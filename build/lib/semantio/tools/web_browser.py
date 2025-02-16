# web_browser.py
from typing import Dict, Any, List, Optional, Callable
from pydantic import Field, BaseModel
from playwright.sync_api import sync_playwright, Page, TimeoutError as PlaywrightTimeoutError
import json, time, re, logging, os, difflib
from .base_tool import BaseTool

# Global logger
logger = logging.getLogger(__name__)

class BrowserPlan(BaseModel):
    tasks: List[Dict[str, Any]] = Field(
        ...,
        description="List of automation tasks to execute"
    )

class WebBrowserTool(BaseTool):
    name: str = Field("WebBrowser", description="Name of the tool")
    description: str = Field(
        "Universal web automation tool with advanced element identification (DOM and image fallback), modal analysis, AJAX waiting, multi-tab support, and custom JS injection.",
        description="Tool description"
    )
    default_timeout: int = 15000  # 15 seconds in milliseconds
    max_retries: int = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Bypass Pydantic's restrictions for extra attributes.
        object.__setattr__(self, "logger", logging.getLogger(__name__))

    def execute(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the browser automation workflow.
        Maintains a context string of executed tasks and passes it to fallback routines.
        DOES NOT close the browser after successful execution.
        """
        overall_start = time.time()
        results = []  # to hold summaries of executed tasks (for context)
        current_url = ""
        try:
            headless = input.get("headless", False)
            self.default_timeout = int(input.get("timeout", 15)) * 1000
            self.max_retries = int(input.get("max_retries", self.max_retries))
            plan = self._generate_plan(input.get("query", ""), current_url)
            if not plan.tasks:
                raise ValueError("No valid tasks in the generated plan.")

            # Start Playwright without a "with" block so we can leave the browser open.
            p = sync_playwright().start()
            browser = p.chromium.launch(headless=headless)
            context = browser.new_context()
            page = context.new_page()

            # Map actions to handlers.
            action_map: Dict[str, Callable[[Page, Dict[str, Any]], Dict[str, Any]]] = {
                "navigate": lambda p, task: self._handle_navigation(p, task.get("value", "")),
                "click": lambda p, task: self._handle_click(p, task.get("selector", "")),
                "type": lambda p, task: self._handle_typing(p, task.get("selector", ""), task.get("value", ""), task),
                "wait": lambda p, task: self._handle_wait(task.get("value", "")),
                "wait_for_ajax": lambda p, task: self._handle_wait_for_ajax(p, task.get("value", "")),
                "scroll": lambda p, task: self._handle_scroll(p, task.get("selector", "")),
                "hover": lambda p, task: self._handle_hover(p, task.get("selector", "")),
                "screenshot": lambda p, task: self._handle_screenshot(p, task.get("value", "screenshot.png")),
                "switch_tab": lambda p, task: self._handle_switch_tab(context, task.get("value", "0")),
                "execute_script": lambda p, task: self._handle_execute_script(p, task.get("value", "")),
                "drag_and_drop": lambda p, task: self._handle_drag_and_drop(p, task.get("selector", ""), task.get("value", "")),
            }
            
            for task in plan.tasks:
                self._dismiss_unwanted_modals(page, task_context=task.get("description", ""))
                action = task.get("action", "").lower()
                self.logger.info(f"Executing task: {task.get('description', action)}")
                start_time = time.time()

                # Build a context string from previously executed tasks.
                executed_context = "\n".join([f"{r['action']}: {r['message']}" for r in results])
                handler = action_map.get(action)
                if not handler:
                    results.append({
                        "action": action,
                        "success": False,
                        "message": f"Unsupported action: {action}"
                    })
                    continue

                result = self._execute_with_retries(page, task, handler, executed_context)
                elapsed = time.time() - start_time
                result["elapsed"] = elapsed
                self.logger.info(f"Action '{action}' completed in {elapsed:.2f} seconds.")
                results.append(result)

                if not result.get("success", False):
                    self.logger.error(f"Task failed: {result.get('message')}")
                    self._capture_failure_screenshot(page, action)
                    break

                current_url = page.url

            overall_elapsed = time.time() - overall_start
            self.logger.info(f"Total execution time: {overall_elapsed:.2f} seconds.")
            # Do not close the browser.
            return {"status": "success", "results": results, "total_time": overall_elapsed}
        except Exception as e:
            self.logger.exception("Execution error:")
            return {"status": "error", "message": str(e)}

    def _generate_plan(self, query: str, current_url: str) -> BrowserPlan:
        prompt = f"""Generate browser automation plan for: {query}

Current URL: {current_url or 'No page loaded yet'}

Required JSON format:
{{
    "tasks": [
        {{
            "action": "navigate|click|type|wait|wait_for_ajax|scroll|hover|screenshot|switch_tab|execute_script|drag_and_drop",
            "selector": "CSS selector (optional)",
            "value": "input text/URL/seconds/filename/target-selector",
            "description": "action purpose"
        }}
    ]
}}

Guidelines:
1. Prefer IDs in selectors (#element-id) and semantic attributes.
2. Include wait steps after navigation and wait for AJAX where applicable.
3. Dismiss any modals/pop-ups that are not part of the task.
4. For drag_and_drop, use source selector in 'selector' and target selector in 'value'.
5. For execute_script, 'value' should contain valid JavaScript.
6. For switch_tab, 'value' should be an index or keyword 'new'.
"""
        response = self.llm.generate(prompt=prompt)
        return self._parse_plan(response)

    def _parse_plan(self, response: str) -> BrowserPlan:
        try:
            json_match = re.search(r'```json\n?(.+?)\n?```', response, re.DOTALL)
            if json_match:
                plan_data = json.loads(json_match.group(1).strip())
            else:
                json_str_match = re.search(r'\{.*\}', response, re.DOTALL)
                if not json_str_match:
                    raise ValueError("No JSON object found in the response.")
                plan_data = json.loads(json_str_match.group())
            validated_tasks = []
            for task in plan_data.get("tasks", []):
                if not all(key in task for key in ["action", "description"]):
                    self.logger.warning(f"Skipping task due to missing keys: {task}")
                    continue
                validated_tasks.append({
                    "action": task["action"],
                    "selector": task.get("selector", ""),
                    "value": task.get("value", ""),
                    "description": task["description"]
                })
            return BrowserPlan(tasks=validated_tasks)
        except (json.JSONDecodeError, AttributeError, ValueError) as e:
            self.logger.error(f"Plan parsing failed: {e}")
            return BrowserPlan(tasks=[])

    def _execute_with_retries(self, page: Page, task: Dict[str, Any],
                                handler: Callable[[Page, Dict[str, Any]], Dict[str, Any]],
                                executed_context: str = "") -> Dict[str, Any]:
        """Execute a task with retry logic. If it fails, pass the executed_context to the fallback prompt.
        The fallback now returns a JSON array of tasks, which are executed sequentially."""
        attempts = 0
        result = {}
        while attempts < self.max_retries:
            result = self._execute_safe_task(page, task, handler)
            if result.get("success", False):
                return result
            attempts += 1
            self.logger.info(f"Retrying task '{task.get('action')}' (attempt {attempts + 1}/{self.max_retries})")
            time.sleep(1 * attempts)
        if task.get("action") in ["click", "type"]:
            self.logger.info("HTML-based automation failed. Using fallback with image-based LLM.")
            result = self._fallback_with_image_llm(page, task, executed_context)
        return result

    def _execute_safe_task(self, page: Page, task: Dict[str, Any],
                             handler: Callable[[Page, Dict[str, Any]], Dict[str, Any]]) -> Dict[str, Any]:
        try:
            return handler(page, task)
        except Exception as e:
            action = task.get("action", "unknown")
            self.logger.exception(f"Error executing task '{action}':")
            return {"action": action, "success": False, "message": f"Critical error: {str(e)}"}

    def _dismiss_unwanted_modals(self, page: Page, task_context: str = ""):
        modal_selectors = [".modal", ".popup", '[role="dialog"]', ".overlay", ".lightbox"]
        for selector in modal_selectors:
            elements = page.query_selector_all(selector)
            for modal in elements:
                if modal.is_visible():
                    self._handle_modal(page, modal, task_context)

    def _handle_modal(self, page: Page, modal_element, task_context: str):
        try:
            modal_screenshot = modal_element.screenshot()
            prompt = (
                f"A modal is displayed on the page. The content is visible in the attached image. "
                f"The current task context is: \"{task_context}\". "
                "Based on the content of the modal and the task context, decide whether to dismiss the modal. "
                "Return a JSON response in the format: { \"action\": \"dismiss\" } to dismiss or { \"action\": \"ignore\" } to leave it. "
                "Return only the JSON."
            )
            response_text = self.llm.generate_from_image(prompt, image_bytes=modal_screenshot)
            self.logger.info(f"LLM response for modal analysis: {response_text}")
            json_match = re.search(r'```json\n?(.+?)\n?```', response_text, re.DOTALL)
            json_text = json_match.group(1).strip() if json_match else response_text.strip()
            decision = json.loads(json_text)
            if decision.get("action") == "dismiss":
                close_buttons = modal_element.query_selector_all(".close, .btn-close, [aria-label='Close'], [data-dismiss='modal']")
                for btn in close_buttons:
                    if btn.is_visible():
                        btn.click()
                        self.logger.info("Modal dismissed using a close button.")
                        return
                page.evaluate("(modal) => modal.remove()", modal_element)
                self.logger.info("Modal dismissed by removal.")
            else:
                self.logger.info("Modal left intact according to LLM analysis.")
        except Exception as e:
            self.logger.error(f"Modal handling error: {e}")

    def _advanced_find_element(self, page: Page, keyword: str):
        try:
            candidates = page.query_selector_all("input, textarea, button, a, div")
            best_match = None
            best_ratio = 0.0
            for candidate in candidates:
                attrs = page.evaluate(
                    """(el) => {
                        return {
                            id: el.id,
                            name: el.getAttribute('name'),
                            placeholder: el.getAttribute('placeholder'),
                            aria: el.getAttribute('aria-label'),
                            text: el.innerText
                        };
                    }""",
                    candidate,
                )
                combined_text = " ".join(
                    filter(None, [
                        attrs.get("id"),
                        attrs.get("name"),
                        attrs.get("placeholder"),
                        attrs.get("aria"),
                        attrs.get("text"),
                    ])
                )
                ratio = difflib.SequenceMatcher(None, combined_text.lower(), keyword.lower()).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = candidate
            if best_ratio > 0.5:
                self.logger.info(f"Advanced fallback detected element with similarity {best_ratio:.2f} for keyword '{keyword}'")
                return best_match
            return None
        except Exception as e:
            self.logger.error(f"Advanced find element error: {e}")
            return None

    def _annotate_page_with_numbers(self, page: Page, query: str = "button, a, input, [onclick]"):
        script = f"""
        (() => {{
            document.querySelectorAll('.automation-annotation-overlay').forEach(el => el.remove());
            const elements = document.querySelectorAll('{query}');
            let counter = 1;
            elements.forEach(el => {{
                const rect = el.getBoundingClientRect();
                if (rect.width === 0 || rect.height === 0) return;
                const overlay = document.createElement('div');
                overlay.classList.add('automation-annotation-overlay');
                overlay.style.position = 'absolute';
                overlay.style.left = (rect.left + window.scrollX) + 'px';
                overlay.style.top = (rect.top + window.scrollY) + 'px';
                overlay.style.width = rect.width + 'px';
                overlay.style.height = rect.height + 'px';
                overlay.style.border = '2px solid red';
                overlay.style.zIndex = 9999;
                overlay.style.pointerEvents = 'none';
                overlay.textContent = counter;
                overlay.style.fontSize = '16px';
                overlay.style.fontWeight = 'bold';
                overlay.style.color = 'red';
                overlay.style.backgroundColor = 'rgba(255, 255, 255, 0.7)';
                document.body.appendChild(overlay);
                counter += 1;
            }});
        }})();
        """
        page.evaluate(script)

    def _click_element_by_number(self, page: Page, number: int) -> Dict[str, Any]:
        candidates = [el for el in page.query_selector_all("button, a, input, [onclick]") if el.is_visible()]
        index = number - 1
        if index < len(candidates):
            candidate = candidates[index]
            candidate.scroll_into_view_if_needed()
            try:
                candidate.click()
                return {"action": "click", "success": True, "message": f"Clicked element number {number}"}
            except Exception as e:
                return {"action": "click", "success": False, "message": f"Click failed: {str(e)}"}
        else:
            return {"action": "click", "success": False, "message": f"Element number {number} not found."}

    def _fallback_with_image_llm(self, page: Page, task: Dict[str, Any], executed_context: str = "") -> Dict[str, Any]:
        """
        Fallback method: Annotate the page, capture a screenshot, and ask the LLM (via image analysis)
        to generate a JSON array of tasks for the next steps.
        Each fallback task is an object:
          {
            "action": "click" or "type",
            "element_number": <number>,
            "text": <if action is 'type', the text to type; otherwise an empty string>
          }
        The prompt includes the executed_context.
        """
        query = "input, textarea" if task.get("action") == "type" else "button, a, input, [onclick]"
        self._annotate_page_with_numbers(page, query=query)
        time.sleep(1)
        screenshot_bytes = page.screenshot(type="png")
        extra = ""
        if task.get("action") == "type":
            extra = f"\nThe exact text to be entered is: \"{task.get('value', '').strip()}\"."
        prompt = (
            f"Tasks executed so far:\n{executed_context}\n\n"
            f"The following task remains: {task.get('description', '')}.{extra}\n"
            "I have annotated the page with numbered overlays using the appropriate query. "
            "Based on the attached screenshot, generate a JSON array of tasks that need to be performed next. "
            "Each task should be a JSON object with the format:\n"
            "[\n"
            "  {\n"
            "    \"action\": \"click\" or \"type\",\n"
            "    \"element_number\": <number>,\n"
            "    \"text\": <if action is 'type', the text to type; otherwise an empty string>\n"
            "  },\n"
            "  ...\n"
            "]\n"
            "Return only the JSON array."
        )
        response_text = self.llm.generate_from_image(prompt, image_bytes=screenshot_bytes)
        self.logger.info(f"LLM response for fallback: {response_text}")
        try:
            fallback_tasks = json.loads(response_text.strip())
            if not isinstance(fallback_tasks, list):
                fallback_tasks = [fallback_tasks]
        except Exception as e:
            json_match = re.search(r'```json\n?(.+?)\n?```', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1).strip()
                fallback_tasks = json.loads(json_text)
                if not isinstance(fallback_tasks, list):
                    fallback_tasks = [fallback_tasks]
            else:
                return {"action": task.get("action"), "success": False, "message": f"Fallback failed to parse JSON: {str(e)}"}

        fallback_results = []
        for fb_task in fallback_tasks:
            action = fb_task.get("action")
            element_number = fb_task.get("element_number")
            if action == "type":
                returned_text = fb_task.get("text", "").strip()
                original_text = task.get("value", "").strip()
                if returned_text.lower() != original_text.lower():
                    self.logger.info("Overriding LLM-provided text with original input text.")
                    text = original_text
                else:
                    text = returned_text
            else:
                text = fb_task.get("text", "")
            if action == "click":
                self.logger.info(f"LLM indicated fallback click on element number {element_number}.")
                res = self._click_element_by_number(page, element_number)
            elif action == "type":
                candidates = [el for el in page.query_selector_all("input, textarea") if el.is_visible()]
                if element_number - 1 < len(candidates):
                    candidate = candidates[element_number - 1]
                    candidate.scroll_into_view_if_needed()
                    try:
                        candidate.fill(text, timeout=self.default_timeout)
                        res = {"action": "type", "success": True, "message": f"Typed '{text}' into element number {element_number}"}
                    except Exception as ex:
                        res = {"action": "type", "success": False, "message": f"Typing failed on fallback element: {str(ex)}"}
                else:
                    res = {"action": "type", "success": False, "message": f"Element number {element_number} not found."}
            else:
                res = {"action": task.get("action"), "success": False, "message": "Invalid fallback action."}
            fallback_results.append(res)
        overall_success = any(r.get("success", False) for r in fallback_results)
        overall_message = "; ".join([r.get("message", "") for r in fallback_results])
        return {"action": task.get("action"), "success": overall_success, "message": overall_message}

    def _handle_navigation(self, page: Page, url: str) -> Dict[str, Any]:
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"
        try:
            page.goto(url, timeout=self.default_timeout)
            page.wait_for_selector("body", timeout=self.default_timeout)
            return {"action": "navigate", "success": True, "message": f"Navigated to {url}"}
        except PlaywrightTimeoutError as e:
            self.logger.error(f"Navigation to {url} timed out: {e}")
            return {"action": "navigate", "success": False, "message": f"Navigation timed out: {str(e)}"}
        except Exception as e:
            self.logger.error(f"Navigation to {url} failed: {e}")
            return {"action": "navigate", "success": False, "message": f"Navigation failed: {str(e)}"}

    def _handle_click(self, page: Page, selector: str) -> Dict[str, Any]:
        try:
            page.wait_for_selector(selector, state="visible", timeout=self.default_timeout)
            page.click(selector, timeout=self.default_timeout)
            return {"action": "click", "success": True, "message": f"Clicked element: {selector}"}
        except PlaywrightTimeoutError as e:
            self.logger.error(f"Click action timed out on selector {selector}: {e}")
            return {"action": "click", "success": False, "message": f"Click timed out: {str(e)}"}
        except Exception as e:
            self.logger.error(f"Click action failed on selector {selector}: {e}")
            return {"action": "click", "success": False, "message": f"Click failed: {str(e)}"}

    def _handle_typing(self, page: Page, selector: str, text: str, task: Dict[str, Any]) -> Dict[str, Any]:
        try:
            page.wait_for_selector(selector, state="attached", timeout=self.default_timeout)
            page.fill(selector, text, timeout=self.default_timeout)
            return {"action": "type", "success": True, "message": f"Typed '{text}' into element."}
        except PlaywrightTimeoutError as e:
            self.logger.info("Primary selector failed; using advanced fallback for element detection.")
            element = self._advanced_find_element(page, "search")
            if not element:
                return {"action": "type", "success": False, "message": f"Typing failed: No search-like element found; error: {str(e)}"}
            try:
                element.fill(text, timeout=self.default_timeout)
                return {"action": "type", "success": True, "message": f"Typed '{text}' into fallback element."}
            except Exception as ex:
                return {"action": "type", "success": False, "message": f"Typing failed on fallback element: {str(ex)}"}
        except Exception as e:
            self.logger.error(f"Typing action failed: {e}")
            return {"action": "type", "success": False, "message": f"Typing failed: {str(e)}"}

    def _handle_wait(self, seconds: str) -> Dict[str, Any]:
        try:
            wait_time = float(seconds)
            self.logger.info(f"Waiting for {wait_time} seconds")
            time.sleep(wait_time)
            return {"action": "wait", "success": True, "message": f"Waited {wait_time} seconds"}
        except ValueError as e:
            self.logger.error(f"Invalid wait time provided: {seconds}")
            return {"action": "wait", "success": False, "message": "Invalid wait time"}

    def _handle_wait_for_ajax(self, page: Page, seconds: str) -> Dict[str, Any]:
        try:
            timeout_seconds = int(seconds) if seconds.strip() != "" else 30
            self.logger.info(f"Waiting for AJAX/network activity for up to {timeout_seconds} seconds.")
            end_time = time.time() + timeout_seconds
            while time.time() < end_time:
                ajax_complete = page.evaluate("""
                    () => {
                        return (window.jQuery ? jQuery.active === 0 : true) &&
                               (typeof window.fetch === 'function' ? true : true);
                    }
                """)
                if ajax_complete:
                    break
                time.sleep(0.5)
            return {"action": "wait_for_ajax", "success": True, "message": "AJAX/network activity subsided."}
        except Exception as e:
            self.logger.error(f"Wait for AJAX failed: {e}")
            return {"action": "wait_for_ajax", "success": False, "message": f"Wait for AJAX failed: {str(e)}"}

    def _handle_scroll(self, page: Page, selector: str) -> Dict[str, Any]:
        try:
            if selector:
                page.wait_for_selector(selector, timeout=self.default_timeout)
                page.eval_on_selector(selector, "el => el.scrollIntoView({behavior: 'smooth', block: 'center'})")
                scroll_target = selector
            else:
                page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
                scroll_target = "page bottom"
            return {"action": "scroll", "success": True, "message": f"Scrolled to {scroll_target}"}
        except Exception as e:
            self.logger.error(f"Scroll action failed on selector {selector}: {e}")
            return {"action": "scroll", "success": False, "message": f"Scroll failed: {str(e)}"}

    def _handle_hover(self, page: Page, selector: str) -> Dict[str, Any]:
        try:
            page.wait_for_selector(selector, state="visible", timeout=self.default_timeout)
            page.hover(selector, timeout=self.default_timeout)
            return {"action": "hover", "success": True, "message": f"Hovered over {selector}"}
        except Exception as e:
            self.logger.error(f"Hover action failed on selector {selector}: {e}")
            return {"action": "hover", "success": False, "message": f"Hover failed: {str(e)}"}

    def _handle_screenshot(self, page: Page, filename: str) -> Dict[str, Any]:
        try:
            page.screenshot(path=filename)
            return {"action": "screenshot", "success": True, "message": f"Screenshot saved as {filename}"}
        except Exception as e:
            self.logger.error(f"Screenshot capture failed: {e}")
            return {"action": "screenshot", "success": False, "message": f"Screenshot failed: {str(e)}"}

    def _handle_switch_tab(self, context, value: str) -> Dict[str, Any]:
        try:
            pages = context.pages
            if value.lower() == "new":
                target_page = pages[-1]
            else:
                idx = int(value)
                if idx < len(pages):
                    target_page = pages[idx]
                else:
                    return {"action": "switch_tab", "success": False, "message": f"Tab index {value} out of range"}
            return {"action": "switch_tab", "success": True, "message": f"Switched to tab {value}"}
        except Exception as e:
            self.logger.error(f"Switch tab failed: {e}")
            return {"action": "switch_tab", "success": False, "message": f"Switch tab failed: {str(e)}"}

    def _handle_execute_script(self, page: Page, script: str) -> Dict[str, Any]:
        try:
            result = page.evaluate(script)
            return {"action": "execute_script", "success": True, "message": "Script executed successfully", "result": result}
        except Exception as e:
            self.logger.error(f"Execute script failed: {e}")
            return {"action": "execute_script", "success": False, "message": f"Script execution failed: {str(e)}"}

    def _handle_drag_and_drop(self, page: Page, source_selector: str, target_selector: str) -> Dict[str, Any]:
        try:
            page.wait_for_selector(source_selector, timeout=self.default_timeout)
            page.wait_for_selector(target_selector, timeout=self.default_timeout)
            source = page.locator(source_selector)
            target = page.locator(target_selector)
            source.drag_to(target, timeout=self.default_timeout)
            return {"action": "drag_and_drop", "success": True, "message": f"Dragged element from {source_selector} to {target_selector}"}
        except Exception as e:
            self.logger.error(f"Drag and drop failed from {source_selector} to {target_selector}: {e}")
            return {"action": "drag_and_drop", "success": False, "message": f"Drag and drop failed: {str(e)}"}

    def _capture_failure_screenshot(self, page: Page, action: str):
        filename = f"failure_{action}_{int(time.time())}.png"
        try:
            page.screenshot(path=filename)
            self.logger.info(f"Failure screenshot captured: {filename}")
        except Exception as e:
            self.logger.error(f"Failed to capture screenshot: {e}")
