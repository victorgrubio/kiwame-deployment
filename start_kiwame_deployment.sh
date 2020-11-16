#!/bin/bash
mkdir db
docker login
docker-compose pull
docker-compose -p kiwame down
docker-compose -p kiwame up -d --no-build  zookeeper kafka mongo api kafdrop
sleep 10
docker-compose -p kiwame up -d cvm dam
