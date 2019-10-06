from django.conf.urls import url
from . import views
from django.contrib.auth import views as auth_views


urlpatterns = {
    url(r'main', views.main, name = 'main'),
    url(r'signup', views.signup, name = 'signup'),
    url(r'registration/login', auth_views.LoginView.as_view(), name = 'login'),
    url(r'registration/logout', auth_views.LogoutView.as_view(next_page='main'), name = 'logout'),
    url(r'profile', views.profile, name = 'profile'),
    url(r'bbs', views.bbs, name = 'bbs'),

}