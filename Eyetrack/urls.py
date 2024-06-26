from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from .views import start_gaze_tracking_view, stop_gaze_tracking_view

urlpatterns = [
    path('start/<int:user_id>/<int:interview_id>/<int:question_id>/', start_gaze_tracking_view, name='start_gaze_tracking'),
    path('stop/<int:user_id>/<int:interview_id>/<int:question_id>/', stop_gaze_tracking_view, name='stop-gaze-tracking'),

]
