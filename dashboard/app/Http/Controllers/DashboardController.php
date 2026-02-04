<?php

namespace App\Http\Controllers;

use App\Models\ActivityLog;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;
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

    public function controlDevice(Request $request)
    {
        // 1. Validasi Input
        $request->validate([
            'device' => 'required|string', // Contoh: D1
            'action' => 'required|string', // Contoh: ON/OFF
        ]);

        $device = $request->input('device');
        $action = $request->input('action');

        // 2. Kirim Perintah ke Python Flask (Raspberry Pi)
        // Kita tembak ke localhost:5000 karena nanti kita set network Laravel ke 'host'
        try {
            $response = Http::timeout(2)->post('http://127.0.0.1:5000/api/manual_command', [
                'device_id' => $device,
                'command' => $action
            ]);

            if ($response->successful()) {
                return back()->with('message', "Berhasil: $device $action");
            } else {
                return back()->withErrors(['error' => 'Gagal menghubungi Hardware (Flask Error)']);
            }
        } catch (\Exception $e) {
            // Fallback jika Python mati: Tetap log manual ke DB biar UI seolah update (Opsional)
            // Tapi sebaiknya return error biar user tau alatnya mati.
            return back()->withErrors(['error' => 'Raspberry Pi tidak merespon!']);
        }
    }
}