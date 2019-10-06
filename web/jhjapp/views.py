from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, get_user
from django.contrib.auth import login as Login
def main(req):
    return render(req, 'jhjapp/main.html', {})
def signup(req):
    #get방식으로 접속
    if req.method == 'GET':
        return render(req, 'jhjapp/signup.html', {'form':UserCreationForm()})
    #post방식으로 접속
    else: #req.method == "POST":
        form = UserCreationForm(req.POST)
        if form.is_valid(): #forn이 유효하면
            form.save() #저장함
            print("Signup Success !")
            return render(req, 'jhjapp/main.html', {}) #메인페이지로 이동
        else: #유효하지않으면
            print("Signup Faield")
            return render(req, 'jhjapp/signup.html', {'form':UserCreationForm()}) #다시 회원가입으로

def login(req):
    if req.method == 'GET':
        return render(req, 'registration/login.html', {})
    else:
        form = authenticate(username=req.username, password=req.password)
        if form:
            Login(req, form)
            print("Login Success !")
            return render(req, 'jhjapp/main.html', {}) #메인페이지로 이동
        else:
            print("Login Failed")
            return render(req, 'registration/login.html', {})  # 다시 회원
def logout(req):
    pass
def profile(req):
    username = get_user(req) #request에서 이름을 받아옴
    usr = User.objects.get(username=username) #db에서 이름에 해당하는 정보를 불러옴
    return render(req, 'jhjapp/profile.html', {'usr':usr}) #넘겨줌
def bbs(req):
    return render(req, 'jhjapp/bbs.html', {})