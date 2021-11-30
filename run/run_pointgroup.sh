#docker build -t pointgroup -f Pointgroup.Dockerfile . 
docker run --networ host \
  --gpus all \
  -v $PWD/PointGroup:/home/docker/PointGroup \
  -it pointgroup
