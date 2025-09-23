#!/bin/bash
echo "🔍 Checking etcd cluster health..."
docker-compose exec etcd etcdctl endpoint health || echo "etcd not running"
