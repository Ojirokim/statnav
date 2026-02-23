from django.urls import path
from . import views

urlpatterns = [
    path("", views.start, name="start"),
    path("wizard/", views.wizard, name="wizard"),
    path("results/", views.results, name="results"),
    path("back/", views.back, name="back"),
    path("lang/<str:lang_code>/", views.switch_lang, name="switch_lang"),
]