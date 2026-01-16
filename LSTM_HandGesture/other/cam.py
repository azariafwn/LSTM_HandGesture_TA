import cv2

def list_ports():
    """
    Mencoba membuka index kamera dari 0 sampai 5
    dan menampilkan feed-nya sebentar agar Anda tahu itu kamera apa.
    """
    print("Mencari kamera yang tersedia (Cek jendela popup)...")
    
    # Cek index 0 sampai 4 (biasanya cukup)
    available_ports = []
    
    for index in range(5):
        print(f"--- Memeriksa Index {index} ---")
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) # CAP_DSHOW agar lebih cepat di Windows
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ DITEMUKAN: Kamera aktif di Index {index}")
                available_ports.append(index)
                
                # Tampilkan nama index di layar video
                cv2.putText(frame, f"Index: {index}", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Tekan Spasi untuk Next", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imshow(f"Cek Kamera Index {index}", frame)
                
                # Tunggu tombol spasi ditekan
                print("   (Tekan Spasi di jendela kamera untuk lanjut cek index berikutnya...)")
                while True:
                    if cv2.waitKey(1) & 0xFF == ord(' '):
                        break
                
                cv2.destroyAllWindows()
            else:
                print(f"⚠️  Index {index} terbuka tapi tidak mengirim gambar (Mungkin kamera virtual/rusak).")
        else:
            print(f"❌ Index {index} tidak ada.")
            
        cap.release()

    print("\n--- HASIL AKHIR ---")
    print(f"Index yang bisa dipakai: {available_ports}")
    print("Gunakan salah satu angka di atas untuk 'cv2.VideoCapture(angka)'")

if __name__ == '__main__':
    list_ports()