find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf
rm -rf gmail-use.zip
zip -r gmail-use.zip app Dockerfile requirements.txt system_prompt.txt server.py requirements.base.txt scripts