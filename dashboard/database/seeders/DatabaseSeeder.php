<?php

namespace Database\Seeders;

use App\Models\User;
use Illuminate\Database\Seeder;
use Illuminate\Support\Facades\Hash;

class DatabaseSeeder extends Seeder
{
    public function run(): void
    {
        // Cek apakah user sudah ada biar ga error kalau dijalankan 2x
        if (!User::where('email', 'admin@admin.com')->exists()) {
            User::factory()->create([
                'name' => 'Admin Smart Home',
                'email' => 'admin@admin.com',
                'password' => Hash::make('password'), // Passwordnya 'password'
            ]);
            echo "✅ User Admin Berhasil Dibuat!\n";
        } else {
            echo "ℹ️ User Admin Sudah Ada.\n";
        }
    }
}