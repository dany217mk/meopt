from functools import wraps

from django.contrib import messages
from django.shortcuts import redirect


def role_required(*roles):
    def decorator(view_func):
        @wraps(view_func)
        def _wrapped(request, *args, **kwargs):
            if not request.user.is_authenticated:
                return redirect("login")
            if request.user.role not in roles:
                messages.error(request, "Недостаточно прав доступа.")
                return redirect("dashboard")
            return view_func(request, *args, **kwargs)

        return _wrapped

    return decorator


def operator_required(view_func):
    @wraps(view_func)
    def _wrapped(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect("login")
        if not request.user.is_staff and request.user.role != "operator":
            messages.error(request, "Недостаточно прав доступа.")
            return redirect("dashboard")
        return view_func(request, *args, **kwargs)

    return _wrapped
