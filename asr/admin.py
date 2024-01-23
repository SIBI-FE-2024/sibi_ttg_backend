from django.contrib import admin
from django_q.models import Task
from . import models


class SubtitleAdmin(admin.ModelAdmin):
    list_display = ("index", "url", "subtitle", "asrtype", "duration")

    def has_add_permission(self, request):
        return True
    
    def has_change_permission(self, request):
        return True


admin.site.register(models.Subtitle, SubtitleAdmin)

