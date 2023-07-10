from django.shortcuts import render

# Create your views here.

def login_request(request):
    return render(request, 'common/login.html')


def register_request(request):
    return render(request, 'common/register.html')