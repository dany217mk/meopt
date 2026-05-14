from django.contrib import messages
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render

from .decorators import operator_required, role_required
from .forms import LoginForm, RegisterForm, RequestForm
from .models import Request, User


def index(request):
    if not request.user.is_authenticated:
        return redirect("login")
    return redirect("dashboard")


@login_required
def dashboard(request):
    if request.user.is_staff or request.user.role == User.ROLE_OPERATOR:
        return redirect("operator_dashboard")
    if request.user.role == User.ROLE_TEAM:
        return redirect("team_dashboard")
    return redirect("client_dashboard")


def login_view(request):
    if request.user.is_authenticated:
        return redirect("dashboard")
    if request.method == "POST":
        form = LoginForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data["email"].strip().lower()
            password = form.cleaned_data["password"]
            user = User.objects.filter(email__iexact=email).first()
            if user:
                authenticated = authenticate(
                    request, username=user.username, password=password
                )
            else:
                authenticated = None
            if authenticated:
                login(request, authenticated)
                return redirect("dashboard")
            messages.error(request, "Неверный email или пароль.")
    else:
        form = LoginForm()
    return render(request, "login.html", {"form": form})


def register_view(request):
    if request.user.is_authenticated:
        return redirect("dashboard")
    if request.method == "POST":
        form = RegisterForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "Регистрация успешна. Войдите в систему.")
            return redirect("login")
    else:
        form = RegisterForm()
    return render(request, "register.html", {"form": form})


@role_required(User.ROLE_CLIENT)
def client_dashboard(request):
    requests = (
        Request.objects.filter(client=request.user)
        .select_related("assigned_team")
        .order_by("-created_at")
    )
    return render(request, "client_dashboard.html", {"requests": requests})


@role_required(User.ROLE_CLIENT)
def create_request(request):
    if request.method == "POST":
        form = RequestForm(request.POST)
        if form.is_valid():
            new_request = form.save(commit=False)
            new_request.client = request.user
            new_request.status = Request.STATUS_NEW
            new_request.save()
            messages.success(request, "Заявка создана.")
            return redirect("client_dashboard")
    else:
        form = RequestForm()
    return render(request, "request_form.html", {"form": form})


@role_required(User.ROLE_CLIENT)
def cancel_request(request, pk):
    if request.method != "POST":
        return JsonResponse({"ok": False, "error": "Недопустимый метод."}, status=405)
    req = get_object_or_404(Request, pk=pk, client=request.user)
    if req.status != Request.STATUS_NEW:
        return JsonResponse(
            {"ok": False, "error": "Заявка уже принята и не может быть отменена."},
            status=400,
        )
    req.status = Request.STATUS_CANCELLED
    req.save(update_fields=["status", "updated_at"])
    messages.success(request, "Заявка отменена.")
    return JsonResponse({"ok": True, "status": req.status, "status_display": req.get_status_display()})


def _team_has_active_request(team):
    return Request.objects.filter(
        assigned_team=team, status__in=[Request.STATUS_ACCEPTED, Request.STATUS_EN_ROUTE]
    ).exists()


@role_required(User.ROLE_TEAM)
def team_dashboard(request):
    active_request = (
        Request.objects.filter(
            assigned_team=request.user,
            status__in=[Request.STATUS_ACCEPTED, Request.STATUS_EN_ROUTE],
        )
        .select_related("client")
        .first()
    )
    new_requests = (
        Request.objects.filter(status=Request.STATUS_NEW, assigned_team__isnull=True)
        .select_related("client")
        .order_by("created_at")
    )
    return render(
        request,
        "team_dashboard.html",
        {"active_request": active_request, "new_requests": new_requests},
    )


@role_required(User.ROLE_TEAM)
@transaction.atomic
def team_accept_request(request, pk):
    if request.method != "POST":
        return JsonResponse({"ok": False, "error": "Недопустимый метод."}, status=405)
    if _team_has_active_request(request.user):
        return JsonResponse(
            {"ok": False, "error": "У вас уже есть активная заявка."}, status=400
        )
    req = get_object_or_404(
        Request, pk=pk, status=Request.STATUS_NEW, assigned_team__isnull=True
    )
    req.assigned_team = request.user
    req.status = Request.STATUS_ACCEPTED
    req.save(update_fields=["assigned_team", "status", "updated_at"])
    messages.success(request, "Заявка принята.")
    return JsonResponse(
        {
            "ok": True,
            "active_request": {
                "id": req.id,
                "address": req.address,
                "description": req.description,
                "status": req.status,
                "status_display": req.get_status_display(),
            },
        }
    )


@role_required(User.ROLE_TEAM)
def team_update_status(request, pk):
    if request.method != "POST":
        return JsonResponse({"ok": False, "error": "Недопустимый метод."}, status=405)
    req = get_object_or_404(Request, pk=pk, assigned_team=request.user)
    next_status = request.POST.get("status")
    allowed_transitions = {
        Request.STATUS_ACCEPTED: Request.STATUS_EN_ROUTE,
        Request.STATUS_EN_ROUTE: Request.STATUS_COMPLETED,
    }
    if allowed_transitions.get(req.status) != next_status:
        return JsonResponse({"ok": False, "error": "Недопустимый переход."}, status=400)
    req.status = next_status
    req.save(update_fields=["status", "updated_at"])
    messages.success(request, "Статус обновлён.")
    return JsonResponse(
        {
            "ok": True,
            "status": req.status,
            "status_display": req.get_status_display(),
            "completed": req.status == Request.STATUS_COMPLETED,
        }
    )


@operator_required
def operator_dashboard(request):
    status_filter = request.GET.get("status")
    requests_qs = Request.objects.select_related("client", "assigned_team").order_by(
        "-created_at"
    )
    if status_filter in dict(Request.STATUS_CHOICES):
        requests_qs = requests_qs.filter(status=status_filter)
    teams = User.objects.filter(role=User.ROLE_TEAM).order_by("team_name", "email")
    busy_team_ids = set(
        Request.objects.filter(
            assigned_team__isnull=False,
            status__in=[Request.STATUS_ACCEPTED, Request.STATUS_EN_ROUTE],
        ).values_list("assigned_team_id", flat=True)
    )
    team_activity = []
    for team in teams:
        active = Request.objects.filter(
            assigned_team=team,
            status__in=[Request.STATUS_ACCEPTED, Request.STATUS_EN_ROUTE],
        ).first()
        team_activity.append({"team": team, "active_request": active})
    context = {
        "requests": requests_qs,
        "teams": teams,
        "busy_team_ids": busy_team_ids,
        "team_activity": team_activity,
        "status_filter": status_filter or "",
        "status_choices": Request.STATUS_CHOICES,
    }
    return render(request, "operator_dashboard.html", context)


@operator_required
@transaction.atomic
def operator_assign(request):
    if request.method != "POST":
        return redirect("operator_dashboard")
    request_id = request.POST.get("request_id")
    team_id = request.POST.get("team_id")
    req = get_object_or_404(Request, pk=request_id)
    team = get_object_or_404(User, pk=team_id, role=User.ROLE_TEAM)
    if req.status in [Request.STATUS_COMPLETED, Request.STATUS_CANCELLED]:
        messages.error(request, "Эту заявку нельзя назначить.")
        return redirect("operator_dashboard")
    if _team_has_active_request(team):
        messages.error(request, "Группа уже занята другой заявкой.")
        return redirect("operator_dashboard")
    req.assigned_team = team
    if req.status == Request.STATUS_NEW:
        req.status = Request.STATUS_ACCEPTED
    req.save(update_fields=["assigned_team", "status", "updated_at"])
    messages.success(request, "Заявка назначена группе.")
    return redirect("operator_dashboard")
