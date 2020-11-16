#!/bin/bash
docker login
docker-compose -f docker-compose-dev.yml pull
docker-compose -f docker-compose-dev.yml -p kiwame_dev down
docker-compose -f docker-compose-dev.yml -p kiwame_dev up -d --no-build  zookeeper kafka mongo api kafdrop
sleep 10
docker-compose -f docker-compose-dev.yml -p kiwame_dev up -d cvm dam
