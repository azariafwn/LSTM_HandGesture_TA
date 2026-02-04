<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class ActivityLog extends Model
{
    // Gunakan koneksi kedua (logs_raspi.db)
    protected $connection = 'sqlite_logs';

    // Nama tabel sesuai yang dibuat script Python
    protected $table = 'activity_logs';

    // Matikan timestamp otomatis Laravel (karena Python punya kolom 'timestamp' sendiri)
    public $timestamps = false;

    // Kolom yang boleh dibaca (opsional, tapi aman didefinisikan)
    protected $guarded = [];
}