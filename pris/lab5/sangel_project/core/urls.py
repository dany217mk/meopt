from django.contrib.auth import views as auth_views
from django.urls import path

from . import views


urlpatterns = [
    path("", views.index, name="home"),
    path("login/", views.login_view, name="login"),
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),
    path("register/", views.register_view, name="register"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("client/", views.client_dashboard, name="client_dashboard"),
    path("client/request/new/", views.create_request, name="request_create"),
    path("client/request/<int:pk>/cancel/", views.cancel_request, name="request_cancel"),
    path("team/", views.team_dashboard, name="team_dashboard"),
    path(
        "team/request/<int:pk>/accept/",
        views.team_accept_request,
        name="team_request_accept",
    ),
    path(
        "team/request/<int:pk>/status/",
        views.team_update_status,
        name="team_request_status",
    ),
    path("operator/", views.operator_dashboard, name="operator_dashboard"),
    path("operator/assign/", views.operator_assign, name="operator_assign"),
]
