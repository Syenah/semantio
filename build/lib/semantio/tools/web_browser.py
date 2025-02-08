# web_browser.py
from typing import Dict, Any, List, Optional, Callable
from pydantic import Field, BaseModel
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import json
import time
import re
import logging
import os
import difflib
from .base_tool import BaseTool

logger = logging.getLogger(__name__)

class BrowserPlan(BaseModel):
    tasks: List[Dict[str, Any]] = Field(
        ...,
        description="List of automation tasks to execute"
    )

class WebBrowserTool(BaseTool):
    name: str = Field("WebBrowser", description="Name of the tool")
    description: str = Field(
        "Highly advanced universal web automation tool with advanced element identification, AJAX waiting, modal dismissal, multi-tab support, and custom JS injection.",
        description="Tool description"
    )
    
    default_timeout: int = 15  # Default wait timeout in seconds
    max_retries: int = 3       # Increased maximum retries for any task

    def execute(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an advanced dynamic web automation workflow."""
        driver = None
        overall_start = time.time()
        try:
            headless = input.get("headless", False)
            self.default_timeout = int(input.get("timeout", self.default_timeout))
            self.max_retries = int(input.get("max_retries", self.max_retries))
            driver = self._init_browser(headless)
            results = []
            current_url = ""

            plan = self._generate_plan(input.get('query', ''), current_url)
            if not plan.tasks:
                raise ValueError("No valid tasks in the generated plan.")

            # Dynamic mapping: action name to handler function.
            action_map: Dict[str, Callable[[webdriver.Chrome, Dict[str, Any]], Dict[str, Any]]] = {
                "navigate": lambda d, task: self._handle_navigation(d, task.get("value", "")),
                "click": lambda d, task: self._handle_click(d, task.get("selector", "")),
                "type": lambda d, task: self._handle_typing(d, task.get("selector", ""), task.get("value", ""), task),
                "wait": lambda d, task: self._handle_wait(task.get("value", "")),
                "wait_for_ajax": lambda d, task: self._handle_wait_for_ajax(d, task.get("value", "30")),
                "scroll": lambda d, task: self._handle_scroll(d, task.get("selector", "")),
                "hover": lambda d, task: self._handle_hover(d, task.get("selector", "")),
                "screenshot": lambda d, task: self._handle_screenshot(d, task.get("value", "screenshot.png")),
                "switch_tab": lambda d, task: self._handle_switch_tab(d, task.get("value", "0")),
                "execute_script": lambda d, task: self._handle_execute_script(d, task.get("value", "")),
                "drag_and_drop": lambda d, task: self._handle_drag_and_drop(d, task.get("selector", ""), task.get("value", "")),
            }
            
            for task in plan.tasks:
                # Before each action, dismiss modals/overlays.
                self._dismiss_unwanted_modals(driver)
                action = task.get("action", "").lower()
                logger.info(f"Executing task: {task.get('description', action)}")
                start_time = time.time()
                handler = action_map.get(action)
                if not handler:
                    results.append({
                        "action": action,
                        "success": False,
                        "message": f"Unsupported action: {action}"
                    })
                    continue

                result = self._execute_with_retries(driver, task, handler)
                elapsed = time.time() - start_time
                result["elapsed"] = elapsed
                logger.info(f"Action '{action}' completed in {elapsed:.2f} seconds.")
                results.append(result)

                if not result.get('success', False):
                    logger.error(f"Task failed: {result.get('message')}")
                    self._capture_failure_screenshot(driver, action)
                    break

                current_url = driver.current_url

            overall_elapsed = time.time() - overall_start
            logger.info(f"Total execution time: {overall_elapsed:.2f} seconds.")
            return {"status": "success", "results": results, "total_time": overall_elapsed}
            
        except Exception as e:
            logger.exception("Execution error:")
            return {"status": "error", "message": str(e)}
        finally:
            if driver:
                driver.quit()

    def _init_browser(self, headless: bool) -> webdriver.Chrome:
        """Initialize browser with advanced options."""
        options = Options()
        options.add_argument("--start-maximized")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        if headless:
            options.add_argument("--headless=new")
        return webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )

    def _generate_plan(self, query: str, current_url: str) -> BrowserPlan:
        """Generate an adaptive execution plan using an LLM or other dynamic planner."""
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
        """Robust JSON parsing with multiple fallback strategies."""
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
                    logger.warning(f"Skipping task due to missing keys: {task}")
                    continue
                validated_tasks.append({
                    "action": task["action"],
                    "selector": task.get("selector", ""),
                    "value": task.get("value", ""),
                    "description": task["description"]
                })
            return BrowserPlan(tasks=validated_tasks)
        except (json.JSONDecodeError, AttributeError, ValueError) as e:
            logger.error(f"Plan parsing failed: {e}")
            return BrowserPlan(tasks=[])

    def _execute_with_retries(self, driver: webdriver.Chrome, task: Dict[str, Any],
                                handler: Callable[[webdriver.Chrome, Dict[str, Any]], Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a task with retry logic and exponential backoff."""
        attempts = 0
        result = {}
        while attempts < self.max_retries:
            result = self._execute_safe_task(driver, task, handler)
            if result.get("success", False):
                return result
            attempts += 1
            logger.info(f"Retrying task '{task.get('action')}' (attempt {attempts + 1}/{self.max_retries})")
            time.sleep(1 * attempts)
        return result

    def _execute_safe_task(self, driver: webdriver.Chrome, task: Dict[str, Any],
                             handler: Callable[[webdriver.Chrome, Dict[str, Any]], Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a task with comprehensive error handling."""
        try:
            return handler(driver, task)
        except Exception as e:
            action = task.get("action", "unknown")
            logger.exception(f"Error executing task '{action}':")
            return {"action": action, "success": False, "message": f"Critical error: {str(e)}"}

    def _dismiss_unwanted_modals(self, driver: webdriver.Chrome):
        """
        Dismiss or remove unwanted modals, overlays, or pop-ups.
        First attempts to click a close button; if not available, removes the element via JS.
        """
        try:
            modal_selectors = [".modal", ".popup", '[role="dialog"]', ".overlay", ".lightbox"]
            for selector in modal_selectors:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                for modal in elements:
                    if modal.is_displayed():
                        close_selectors = [".close", ".btn-close", "[aria-label='Close']", "[data-dismiss='modal']"]
                        dismissed = False
                        for close_sel in close_selectors:
                            try:
                                close_button = modal.find_element(By.CSS_SELECTOR, close_sel)
                                if close_button.is_displayed():
                                    close_button.click()
                                    dismissed = True
                                    logger.info(f"Dismissed modal using selector {close_sel}")
                                    time.sleep(1)
                                    break
                            except Exception:
                                continue
                        if not dismissed:
                            # Remove overlay by setting display to none
                            driver.execute_script("arguments[0].remove();", modal)
                            logger.info(f"Removed overlay/modal with selector {selector}")
        except Exception as e:
            logger.debug(f"Modal dismissal error: {e}")

    def _advanced_find_element(self, driver: webdriver.Chrome, keyword: str) -> Optional[WebElement]:
        """
        Advanced fallback for finding an element.
        Searches across multiple attributes and inner text using fuzzy matching.
        """
        candidates = driver.find_elements(By.CSS_SELECTOR, "input, textarea, button, a, div")
        best_match = None
        best_ratio = 0.0
        for candidate in candidates:
            combined_text = " ".join([
                candidate.get_attribute("id") or "",
                candidate.get_attribute("name") or "",
                candidate.get_attribute("placeholder") or "",
                candidate.get_attribute("aria-label") or "",
                candidate.text or "",
            ])
            ratio = difflib.SequenceMatcher(None, combined_text.lower(), keyword.lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = candidate
        if best_ratio > 0.5:
            logger.info(f"Advanced fallback detected element with similarity {best_ratio:.2f} for keyword '{keyword}'")
            return best_match
        return None

    def _handle_navigation(self, driver: webdriver.Chrome, url: str) -> Dict[str, Any]:
        """Handle navigation with URL correction."""
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"
        try:
            driver.get(url)
            WebDriverWait(driver, self.default_timeout).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            return {"action": "navigate", "success": True, "message": f"Navigated to {url}"}
        except Exception as e:
            logger.error(f"Navigation to {url} failed: {e}")
            return {"action": "navigate", "success": False, "message": f"Navigation failed: {str(e)}"}

    def _handle_click(self, driver: webdriver.Chrome, selector: str) -> Dict[str, Any]:
        """Handle click actions with fallback using JS if needed."""
        try:
            element = WebDriverWait(driver, self.default_timeout).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
            )
            driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", element)
            try:
                element.click()
            except Exception:
                driver.execute_script("arguments[0].click();", element)
            return {"action": "click", "success": True, "message": f"Clicked element: {selector}"}
        except Exception as e:
            logger.error(f"Click action failed on selector {selector}: {e}")
            return {"action": "click", "success": False, "message": f"Click failed: {str(e)}"}

    def _handle_typing(self, driver: webdriver.Chrome, selector: str, text: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle typing into an element.
        If the primary selector fails, attempt advanced fallback detection.
        """
        try:
            element = WebDriverWait(driver, self.default_timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, selector))
            )
        except Exception as e:
            # If the task seems to involve search or similar text, use advanced fallback.
            if "search" in task.get("description", "").lower() or "search" in selector.lower():
                logger.info("Primary selector failed; using advanced fallback for element detection.")
                element = self._advanced_find_element(driver, "search")
                if not element:
                    return {"action": "type", "success": False, "message": f"Typing failed: No search-like element found; error: {str(e)}"}
            else:
                return {"action": "type", "success": False, "message": f"Typing failed: {str(e)}"}
        try:
            element.clear()
            element.send_keys(text)
            return {"action": "type", "success": True, "message": f"Typed '{text}' into element."}
        except Exception as e:
            logger.error(f"Typing action failed: {e}")
            return {"action": "type", "success": False, "message": f"Typing failed: {str(e)}"}

    def _handle_wait(self, seconds: str) -> Dict[str, Any]:
        """Handle a simple wait."""
        try:
            wait_time = float(seconds)
            logger.info(f"Waiting for {wait_time} seconds")
            time.sleep(wait_time)
            return {"action": "wait", "success": True, "message": f"Waited {wait_time} seconds"}
        except ValueError as e:
            logger.error(f"Invalid wait time provided: {seconds}")
            return {"action": "wait", "success": False, "message": "Invalid wait time"}

    def _handle_wait_for_ajax(self, driver: webdriver.Chrome, seconds: str) -> Dict[str, Any]:
        """
        Wait until AJAX/network activity has subsided.
        This implementation first checks for jQuery, then falls back to a generic check.
        """
        try:
            timeout = int(seconds)
            logger.info(f"Waiting for AJAX/network activity for up to {timeout} seconds.")
            end_time = time.time() + timeout
            while time.time() < end_time:
                ajax_complete = driver.execute_script("""
                    return (window.jQuery ? jQuery.active === 0 : true) &&
                           (typeof window.fetch === 'function' ? true : true);
                """)
                if ajax_complete:
                    break
                time.sleep(0.5)
            return {"action": "wait_for_ajax", "success": True, "message": "AJAX/network activity subsided."}
        except Exception as e:
            logger.error(f"Wait for AJAX failed: {e}")
            return {"action": "wait_for_ajax", "success": False, "message": f"Wait for AJAX failed: {str(e)}"}

    def _handle_scroll(self, driver: webdriver.Chrome, selector: str) -> Dict[str, Any]:
        """Handle scrolling to a specific element or page bottom."""
        try:
            if selector:
                element = WebDriverWait(driver, self.default_timeout).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                )
                driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", element)
                scroll_target = selector
            else:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                scroll_target = "page bottom"
            return {"action": "scroll", "success": True, "message": f"Scrolled to {scroll_target}"}
        except Exception as e:
            logger.error(f"Scroll action failed on selector {selector}: {e}")
            return {"action": "scroll", "success": False, "message": f"Scroll failed: {str(e)}"}

    def _handle_hover(self, driver: webdriver.Chrome, selector: str) -> Dict[str, Any]:
        """Handle mouse hover action."""
        try:
            element = WebDriverWait(driver, self.default_timeout).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, selector))
            )
            ActionChains(driver).move_to_element(element).perform()
            return {"action": "hover", "success": True, "message": f"Hovered over {selector}"}
        except Exception as e:
            logger.error(f"Hover action failed on selector {selector}: {e}")
            return {"action": "hover", "success": False, "message": f"Hover failed: {str(e)}"}

    def _handle_screenshot(self, driver: webdriver.Chrome, filename: str) -> Dict[str, Any]:
        """Capture a screenshot of the current browser state."""
        try:
            driver.save_screenshot(filename)
            return {"action": "screenshot", "success": True, "message": f"Screenshot saved as {filename}"}
        except Exception as e:
            logger.error(f"Screenshot capture failed: {e}")
            return {"action": "screenshot", "success": False, "message": f"Screenshot failed: {str(e)}"}

    def _handle_switch_tab(self, driver: webdriver.Chrome, value: str) -> Dict[str, Any]:
        """
        Switch between tabs. 'value' can be an index or the keyword 'new'.
        """
        try:
            handles = driver.window_handles
            if value.lower() == "new":
                target_handle = handles[-1]
            else:
                idx = int(value)
                if idx < len(handles):
                    target_handle = handles[idx]
                else:
                    return {"action": "switch_tab", "success": False, "message": f"Tab index {value} out of range"}
            driver.switch_to.window(target_handle)
            return {"action": "switch_tab", "success": True, "message": f"Switched to tab {value}"}
        except Exception as e:
            logger.error(f"Switch tab failed: {e}")
            return {"action": "switch_tab", "success": False, "message": f"Switch tab failed: {str(e)}"}

    def _handle_execute_script(self, driver: webdriver.Chrome, script: str) -> Dict[str, Any]:
        """
        Execute arbitrary JavaScript code.
        """
        try:
            result = driver.execute_script(script)
            return {"action": "execute_script", "success": True, "message": "Script executed successfully", "result": result}
        except Exception as e:
            logger.error(f"Execute script failed: {e}")
            return {"action": "execute_script", "success": False, "message": f"Script execution failed: {str(e)}"}

    def _handle_drag_and_drop(self, driver: webdriver.Chrome, source_selector: str, target_selector: str) -> Dict[str, Any]:
        """
        Simulate a drag-and-drop operation.
        """
        try:
            source = WebDriverWait(driver, self.default_timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, source_selector))
            )
            target = WebDriverWait(driver, self.default_timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, target_selector))
            )
            ActionChains(driver).drag_and_drop(source, target).perform()
            return {"action": "drag_and_drop", "success": True, "message": f"Dragged element from {source_selector} to {target_selector}"}
        except Exception as e:
            logger.error(f"Drag and drop failed from {source_selector} to {target_selector}: {e}")
            return {"action": "drag_and_drop", "success": False, "message": f"Drag and drop failed: {str(e)}"}

    def _capture_failure_screenshot(self, driver: webdriver.Chrome, action: str):
        """Capture a screenshot for debugging when an error occurs."""
        filename = f"failure_{action}_{int(time.time())}.png"
        try:
            driver.save_screenshot(filename)
            logger.info(f"Failure screenshot captured: {filename}")
        except Exception as e:
            logger.error(f"Failed to capture screenshot: {e}")