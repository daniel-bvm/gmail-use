from browser_use import Controller, Browser, ActionResult
from .models import browser_use_custom_models
from pydantic import BaseModel
from browser_use.browser.context import BrowserContext

_controller = Controller(
    output_model=browser_use_custom_models.FinalAgentResult
)

@_controller.registry.action('Sign out all Google accounts')
async def sign_out(browser: BrowserContext):
    page = await browser.get_current_page()

    await page.goto(
        'https://accounts.google.com/signout/chrome/landing', 
        wait_until='documentloaded'
    )

    return ActionResult(extracted_content='Sign out successful!')

def get_controler():
    global _controller
    return _controller