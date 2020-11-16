#!/bin/bash
docker rm -f kiwame_$1_1
docker-compose -f docker-compose-dev.yml -p kiwame up --no-deps --build -d $1
