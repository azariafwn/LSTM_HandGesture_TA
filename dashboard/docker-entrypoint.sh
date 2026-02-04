#!/bin/bash

# 1. Setup .env jika belum ada (Copy dari example)
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env
fi

# 2. Setup Folder Permissions (Biar gak error permission denied)
chmod -R 777 storage bootstrap/cache
chmod -R 777 storage/logs_shared

# 3. Generate Key (Hanya jika belum ada di .env)
# Kita cek apakah APP_KEY masih kosong atau default
if grep -q "APP_KEY=" .env && [ -z "$(grep "APP_KEY=base64" .env)" ]; then
    echo "Generating Application Key..."
    php artisan key:generate
fi

# 4. Setup Database (SQLite)
touch database/database.sqlite
chmod 777 database/database.sqlite

# 5. Migrate & Seed (Pakai --force karena production)
echo "Migrating Database..."
php artisan migrate --force

echo "Seeding Admin User..."
php artisan db:seed --force

# 6. Start Server
echo "Starting Laravel Server..."
php artisan serve --host=0.0.0.0 --port=8000