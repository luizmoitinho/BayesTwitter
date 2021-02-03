@echo off

echo Instalando modulo tweepy ...
pip install tweepy > nul

echo Instalando modulo pandas ...
pip install pandas > nul

echo Instalando modulo imageio ...
pip install imageio > nul

echo Instalando modulo sklearn ...
pip install sklearn > nul

echo Instalando modulo flask ...
pip install flask > nul

echo Instalando modulo flask_cors ...
pip install flask_cors > nul

echo Instalando modulo wordcloud ...
pip install wordcloud > nul

echo Instalando modulo geopy ...
pip install geopy > nul

echo Instalando modulo folium ...
pip install folium > nul

echo Abrindo arquivo web
start index.html > nul

echo Iniciando servidor, aguarde...
python twitter_analysis.py