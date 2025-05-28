from browser_use import Controller, Browser, ActionResult
from .models import browser_use_custom_models
from browser_use.browser.context import BrowserContext
from .signals import (
    UnauthorizedAccess,
    RequireUserConfirmation
)
from typing import Literal
from fnmatch import fnmatch
import logging
from playwright._impl._api_structures import (
    ClientCertificate,
    Cookie
)
from typing import TypedDict

logger = logging.getLogger(__name__)

built_in_actions = [
    'done',
    'search_google',
    'go_to_url',
    'go_back',
    'wait',
    'click_element_by_index',
    'input_text',
    'save_pdf',
    'switch_tab',
    'open_tab',
    'close_tab',
    'extract_content',
    'scroll_down',
    'scroll_up',
    'send_keys',
    'scroll_to_text',
    'get_dropdown_options',
    'select_dropdown_option',
    'drag_drop',
    'get_sheet_contents',
    'select_cell_or_range',
    'get_range_contents',
    'clear_selected_range',
    'input_selected_cell_text',
    'update_range_contents' 
]

exclude = [
    a
    for a in built_in_actions
    if a not in [
        'done',
        # 'search_google',
        'go_to_url',
        'go_back',
        # 'wait',
        'click_element_by_index',
        'input_text',
        # 'save_pdf',
        # 'switch_tab',
        # 'open_tab',
        # 'close_tab',
        'extract_content',
        'scroll_down',
        'scroll_up',
        'send_keys',
        # 'scroll_to_text',

        'get_dropdown_options',
        'select_dropdown_option',

        # 'drag_drop',
        # 'get_sheet_contents',
        # 'select_cell_or_range',
        # 'get_range_contents',
        # 'clear_selected_range',
        # 'input_selected_cell_text',
        'update_range_contents' 
    ]
]

_controller = Controller(
    output_model=browser_use_custom_models.BasicAgentResponse,
    exclude_actions=exclude
)

async def check_authorization(ctx: BrowserContext) -> bool:
    cookies = await ctx.session.context.cookies('https://mail.google.com')

    for cookie in cookies:
        
        name = cookie.get('name', '')
        value = cookie.get('value', '')

        if name == 'SID' and value != '':
            return True

    return False

async def ensure_url(ctx: BrowserContext, url: str) -> None:
    page = await ctx.get_current_page()
    current_url = page.url

    if not fnmatch(current_url, url + '*'):
        logger.info(f'Navigating to {url} from {current_url}')
        await page.goto(url, wait_until='networkidle')
        
    return fnmatch(current_url, url + '*')

async def sign_out(browser: BrowserContext):
    sites = ['accounts.google.com', 'mail.google.com']

    for site in sites:
        await browser.session.context.clear_cookies(domain=site)
        
    page = await browser.get_current_page()
    await page.reload(wait_until='networkidle')

    return ActionResult(extracted_content='Sign out successful!')

# @_controller.action('Open the user mail box')
async def open_mail_box(browser: BrowserContext):
    page = await browser.get_current_page()
    
    await page.goto(
        'https://mail.google.com/mail/u/0/', 
        wait_until='networkidle' # networkidle
    )

    return ActionResult(extracted_content='Navigated to input box')

async def fill_email_form(browser: BrowserContext, subject: str, body: str, recipient: str = None):
    page = await browser.get_current_page()

    if recipient is not None:
        await page.fill('input[aria-label="To recipients"]', recipient)

    await page.fill('input[name="subjectbox"]', subject)
    await page.fill('div[aria-label="Message Body"]', body)

    return ActionResult(extracted_content='Email form filled successfully!')

async def search_email(browser: BrowserContext, query: str):
    page = await browser.get_current_page()

    # Wait for the search box to be available
    await page.wait_for_selector('input[name="q"]', timeout=5000)
    await page.fill('input[name="q"]', query)
    await page.click('button[aria-label="Search mail"]')
    await page.wait_for_timeout(2000)

    return ActionResult(extracted_content='Search executed successfully!')

def get_basic_controler():
    global _controller
