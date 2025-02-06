# web_browser.py
from typing import Dict, Any, List, Optional
from pydantic import Field, BaseModel
from selenium import webdriver
from selenium.webdriver.common.by import By
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
        "Universal web automation tool for dynamic website interactions",
        description="Tool description"
    )
    
    def execute(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute dynamic web automation workflow"""
        driver = None
        try:
            driver = self._init_browser(input.get("headless", False))
            results = []
            current_url = ""

            # Generate initial plan
            plan = self._generate_plan(input['query'], current_url)
            
            for task in plan.tasks:
                result = self._execute_safe_task(driver, task)
                results.append(result)
                
                if not result['success']:
                    break
                    
                # Update context for next tasks
                current_url = driver.current_url

            return {"status": "success", "results": results}
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
        finally:
            if driver:
                driver.quit()

    def _init_browser(self, headless: bool) -> webdriver.Chrome:
        """Initialize browser with advanced options"""
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
        """Generate adaptive execution plan using LLM"""
        prompt = f"""Generate browser automation plan for: {query}
        
        Current URL: {current_url or 'No page loaded yet'}
        
        Required JSON format:
        {{
            "tasks": [
                {{
                    "action": "navigate|click|type|wait|scroll",
                    "selector": "CSS selector (optional)",
                    "value": "input text/URL/seconds",
                    "description": "action purpose"
                }}
            ]
        }}
        
        Guidelines:
        1. Prefer IDs in selectors (#element-id)
        2. Use semantic attributes (aria-label, name)
        3. Include wait steps after navigation
        4. Prioritize visible elements
        5. Add scroll steps for hidden elements
        """
        
        response = self.llm.generate(prompt=prompt)
        return self._parse_plan(response)

    def _parse_plan(self, response: str) -> BrowserPlan:
        """Robust JSON parsing with multiple fallback strategies"""
        try:
            # Try extracting JSON from markdown code block
            json_match = re.search(r'```json\n?(.+?)\n?```', response, re.DOTALL)
            if json_match:
                plan_data = json.loads(json_match.group(1).strip())
            else:
                # Fallback to extract first JSON object
                json_str = re.search(r'\{.*\}', response, re.DOTALL).group()
                plan_data = json.loads(json_str)
            
            # Validate tasks structure
            validated_tasks = []
            for task in plan_data.get("tasks", []):
                if not all(key in task for key in ["action", "description"]):
                    continue
                validated_tasks.append({
                    "action": task["action"],
                    "selector": task.get("selector", ""),
                    "value": task.get("value", ""),
                    "description": task["description"]
                })
            
            return BrowserPlan(tasks=validated_tasks)
            
        except (json.JSONDecodeError, AttributeError) as e:
            logger.error(f"Plan parsing failed: {e}")
            return BrowserPlan(tasks=[])

    def _execute_safe_task(self, driver, task: Dict) -> Dict[str, Any]:
        """Execute task with comprehensive error handling"""
        try:
            action = task["action"].lower()
            selector = task.get("selector", "")
            value = task.get("value", "")
            
            if action == "navigate":
                return self._handle_navigation(driver, value)
                
            elif action == "click":
                return self._handle_click(driver, selector)
                
            elif action == "type":
                return self._handle_typing(driver, selector, value)
                
            elif action == "wait":
                return self._handle_wait(value)
                
            elif action == "scroll":
                return self._handle_scroll(driver, selector)
                
            return {
                "action": action,
                "success": False,
                "message": f"Unsupported action: {action}"
            }
            
        except Exception as e:
            return {
                "action": action,
                "success": False,
                "message": f"Critical error: {str(e)}"
            }

    def _handle_navigation(self, driver, url: str) -> Dict[str, Any]:
        """Smart navigation handler"""
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"
            
        try:
            driver.get(url)
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            return {
                "action": "navigate",
                "success": True,
                "message": f"Navigated to {url}"
            }
        except Exception as e:
            return {
                "action": "navigate",
                "success": False,
                "message": f"Navigation failed: {str(e)}"
            }

    def _handle_click(self, driver, selector: str) -> Dict[str, Any]:
        """Dynamic click handler"""
        try:
            element = WebDriverWait(driver, 15).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
            )
            driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth'});", element)
            element.click()
            return {
                "action": "click",
                "success": True,
                "message": f"Clicked element: {selector}"
            }
        except Exception as e:
            return {
                "action": "click",
                "success": False,
                "message": f"Click failed: {str(e)}"
            }

    def _handle_typing(self, driver, selector: str, text: str) -> Dict[str, Any]:
        """Universal typing handler"""
        try:
            element = WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, selector))
            )
            element.clear()
            element.send_keys(text)
            return {
                "action": "type",
                "success": True,
                "message": f"Typed '{text}' into {selector}"
            }
        except Exception as e:
            return {
                "action": "type",
                "success": False,
                "message": f"Typing failed: {str(e)}"
            }

    def _handle_wait(self, seconds: str) -> Dict[str, Any]:
        """Configurable wait handler"""
        try:
            wait_time = float(seconds)
            time.sleep(wait_time)
            return {
                "action": "wait",
                "success": True,
                "message": f"Waited {wait_time} seconds"
            }
        except ValueError:
            return {
                "action": "wait",
                "success": False,
                "message": "Invalid wait time"
            }

    def _handle_scroll(self, driver, selector: str) -> Dict[str, Any]:
        """Smart scroll handler"""
        try:
            if selector:
                element = WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                )
                driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth'});", element)
            else:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                
            return {
                "action": "scroll",
                "success": True,
                "message": f"Scrolled to {selector or 'page bottom'}"
            }
        except Exception as e:
            return {
                "action": "scroll",
                "success": False,
                "message": f"Scroll failed: {str(e)}"
            }