# xem logs
docker compose -f /home/ubuntu/vinh/noraquantengine/Grey/docker/docker-compose.yml logs -f backend 
# (--tail 120)
docker logs grey-backend --tail 120

# restart
docker compose -f /home/ubuntu/vinh/noraquantengine/Grey/docker/docker-compose.yml restart backend

# stop
docker compose -f /home/ubuntu/vinh/noraquantengine/Grey/docker/docker-compose.yml stop backend
