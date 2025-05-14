find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf
rm -rf haha.zip
zip -r haha.zip app Dockerfile requirements.txt system_prompt.txt server.py