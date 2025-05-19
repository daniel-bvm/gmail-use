from browser_use import Controller, Browser, ActionResult
from .models import browser_use_custom_models
from browser_use.browser.context import BrowserContext
from .signals import (
    UnauthorizedAccess,
    RequireUserConfirmation
)
from typing import Literal
from fnmatch import fnmatch

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

_controller = Controller(
    output_model=browser_use_custom_models.FinalAgentResult,
    exclude_actions=[
        a
        for a in built_in_actions
        if a not in [
            'done',
            # 'search_google',
            'go_to_url',
            'go_back',
            'wait',
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
            # 'get_dropdown_options',
            # 'select_dropdown_option',
            # 'drag_drop',
            # 'get_sheet_contents',
            # 'select_cell_or_range',
            # 'get_range_contents',
            # 'clear_selected_range',
            # 'input_selected_cell_text',
            'update_range_contents' 
        ]
    ]
)


async def check_authorization(ctx: BrowserContext) -> bool:
    site = 'accounts.google.com'
    cookies = await ctx.session.context.cookies()

    for cookie in cookies:
        if not hasattr(cookie, 'domain') or not hasattr(cookie, 'name') or not hasattr(cookie, 'value'):
            continue

        if cookie.domain == site and cookie.name == 'SID' and cookie.value:
            return True

    return False


async def ensure_url(ctx: BrowserContext, pat: str) -> None:
    page = await ctx.get_current_page()
    current_url = page.url

    if not fnmatch(current_url, pat):
        await page.goto(pat, wait_until='networkidle0')

@_controller.action('Sign out Google account')
async def sign_out(browser: BrowserContext):
    page = await browser.get_current_page()

    await page.goto(
        'https://accounts.google.com/signout/chrome/landing', 
        wait_until='networkidle0'
    )

    return ActionResult(extracted_content='Sign out successful!')

@_controller.action('Open the user mail box')
async def open_mail_box(browser: BrowserContext):
    page = await browser.get_current_page()
    
    await page.goto(
        'https://mail.google.com/mail/u/0/', 
        wait_until='networkidle'
    )

    return ActionResult(extracted_content='Navigated to input box')

@_controller.action('Filter mail by criteria')
async def filter_mail(
    browser: BrowserContext, 
    by_sender: str = None, 
    tag: Literal['inbox', 'sent', 'drafts', 'spam', 'trash', 'important', 'primary', 'promotion', 'social', 'unread'] = 'inbox'
):
    page = await browser.get_current_page()

    if by_sender is not None:
        pass

    return ActionResult(extracted_content='Navigated to input box')


@_controller.action('Ask for user confirmation before proceeding')
async def wait_for_user_confirmation(
    message: str,
    browser: BrowserContext
):
    return message

def get_controler():
    global _controller
    return _controller