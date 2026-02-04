import { defineConfig } from 'vite';
import laravel from 'laravel-vite-plugin';
import react from '@vitejs/plugin-react';

export default defineConfig({
    plugins: [
        laravel({
            input: 'resources/js/app.jsx',
            refresh: true,
        }),
        react(),
    ],
    // --- TAMBAHAN KHUSUS DOCKER WINDOWS ---
    server: {
        host: '0.0.0.0', // Biar bisa diakses dari luar container
        port: 5173,      // Port default Vite
        hmr: {
            host: 'localhost', // Hot Module Replacement diarahkan ke Windows
        },
        watch: {
            usePolling: true, // Wajib buat Windows biar deteksi perubahan file
        },
    },
});