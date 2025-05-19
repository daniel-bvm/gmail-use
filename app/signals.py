class UnauthorizedAccess(Exception):
    """Exception raised for unauthorized access attempts."""
    pass

class RequireUserConfirmation(Exception):
    """Exception raised when user confirmation is required."""
    pass