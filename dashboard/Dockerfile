FROM php:8.2-fpm

# 1. Install Extension PHP (Pakai Installer Cepat biar ga nunggu lama)
COPY --from=mlocati/php-extension-installer /usr/bin/install-php-extensions /usr/local/bin/
RUN install-php-extensions pdo_mysql mbstring exif pcntl bcmath gd pdo_sqlite

# 2. Install System Tools
RUN apt-get update && apt-get install -y \
    unzip \
    sqlite3 \
    libsqlite3-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. Install Composer
COPY --from=composer:latest /usr/bin/composer /usr/bin/composer

# 4. Setup Folder Kerja
WORKDIR /var/www

# 5. Copy Semua Kodingan (KECUALI node_modules & vendor)
COPY . /var/www

# 6. Install Dependency Backend (Laravel)
# Kita install di sini biar masuk ke dalam Image
RUN composer install --no-interaction --optimize-autoloader --no-dev

# 7. Setup Permission Folder
RUN chown -R www-data:www-data /var/www \
    && chmod -R 777 storage bootstrap/cache

# 8. Setup Script Startup
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# 9. Expose Port
EXPOSE 8000

# 10. JALANKAN SCRIPT MAGIC SAAT STARTUP
ENTRYPOINT ["docker-entrypoint.sh"]