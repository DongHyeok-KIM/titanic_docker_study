docker build -t titanic-docker
docker run --rm -d --name titanic-docker \
-v /Users/donut/Documents/tmp:/homework/app/data \
 -v /etc/localtime:/etc/localtime:ro \
-p 8000:8010 titanic-docker