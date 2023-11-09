from django.shortcuts import render
from django.http import HttpResponse
from . import main

def home(request):
	return render(request,"chatbot.html")

def chatBotReply(request):
    message = request.GET['message']
    messages = main.finalPredict(message)
    print(messages)
    return HttpResponse(messages)