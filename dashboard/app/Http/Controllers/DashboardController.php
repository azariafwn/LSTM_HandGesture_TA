<?php

namespace App\Http\Controllers;

use App\Models\ActivityLog;
use Illuminate\Http\Request;
use Inertia\Inertia;

class DashboardController extends Controller
{
    public function index()
    {
        // Ambil 20 data terakhir, urutkan dari yang terbaru
        // Kita bungkus try-catch biar kalau DB belum ada isinya, ga error
        try {
            $logs = ActivityLog::orderBy('id', 'desc')->take(20)->get();
        } catch (\Exception $e) {
            $logs = []; // Kalau error/kosong, kirim array kosong
        }

        return Inertia::render('Dashboard', [
            'logs' => $logs
        ]);
    }
}