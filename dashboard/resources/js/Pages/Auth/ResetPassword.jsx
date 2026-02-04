import InputError from '@/Components/InputError';
import InputLabel from '@/Components/InputLabel';
import PrimaryButton from '@/Components/PrimaryButton';
import TextInput from '@/Components/TextInput';
import { Head, useForm } from '@inertiajs/react';

export default function ResetPassword({ token, email }) {
    const { data, setData, post, processing, errors, reset } = useForm({
        token: token,
        email: email,
        password: '',
        password_confirmation: '',
    });

    const submit = (e) => {
        e.preventDefault();

        post(route('password.store'), {
            onFinish: () => reset('password', 'password_confirmation'),
        });
    };

    return (
        // Wrapper Utama (Gaya Cyberpunk/Futuristik)
        <div className="min-h-screen flex flex-col sm:justify-center items-center pt-6 sm:pt-0 bg-slate-950 font-mono text-slate-300">
            <Head title="Reset Password" />

            {/* Header */}
            <div className="mb-6 text-center">
                <h1 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500 tracking-tighter">
                    CREDENTIAL_RESET
                </h1>
                <p className="text-xs text-slate-500 mt-1 tracking-widest">SETTING NEW ACCESS CODE</p>
            </div>

            {/* Card Container */}
            <div className="w-full sm:max-w-md mt-6 px-6 py-8 bg-slate-900/50 border border-slate-800 shadow-[0_0_15px_rgba(0,0,0,0.5)] overflow-hidden sm:rounded-xl backdrop-blur-sm relative group">
                
                {/* Efek Sudut Dekoratif */}
                <div className="absolute top-0 left-0 w-4 h-4 border-t-2 border-l-2 border-cyan-500/50 rounded-tl-lg"></div>
                <div className="absolute top-0 right-0 w-4 h-4 border-t-2 border-r-2 border-cyan-500/50 rounded-tr-lg"></div>
                <div className="absolute bottom-0 left-0 w-4 h-4 border-b-2 border-l-2 border-cyan-500/50 rounded-bl-lg"></div>
                <div className="absolute bottom-0 right-0 w-4 h-4 border-b-2 border-r-2 border-cyan-500/50 rounded-br-lg"></div>

                <form onSubmit={submit}>
                    {/* Email Field (Biasanya Readonly/Hidden tapi kita tampilkan untuk konfirmasi) */}
                    <div>
                        <InputLabel htmlFor="email" value="IDENTITY_KEY (Email)" className="text-slate-400 text-xs tracking-widest mb-1" />

                        <TextInput
                            id="email"
                            type="email"
                            name="email"
                            value={data.email}
                            className="mt-1 block w-full bg-slate-950 border-slate-700 text-slate-200 focus:border-cyan-500 focus:ring-cyan-500/50 placeholder-slate-600 rounded-md"
                            autoComplete="username"
                            onChange={(e) => setData('email', e.target.value)}
                            // Email biasanya readonly di step ini, tapi boleh diedit jika perlu
                        />

                        <InputError message={errors.email} className="mt-2 text-rose-400" />
                    </div>

                    {/* Password Baru */}
                    <div className="mt-4">
                        <InputLabel htmlFor="password" value="NEW_ACCESS_CODE" className="text-slate-400 text-xs tracking-widest mb-1" />

                        <TextInput
                            id="password"
                            type="password"
                            name="password"
                            value={data.password}
                            className="mt-1 block w-full bg-slate-950 border-slate-700 text-slate-200 focus:border-cyan-500 focus:ring-cyan-500/50 placeholder-slate-600 rounded-md"
                            autoComplete="new-password"
                            isFocused={true}
                            onChange={(e) => setData('password', e.target.value)}
                            placeholder="••••••••"
                        />

                        <InputError message={errors.password} className="mt-2 text-rose-400" />
                    </div>

                    {/* Konfirmasi Password */}
                    <div className="mt-4">
                        <InputLabel htmlFor="password_confirmation" value="CONFIRM_NEW_CODE" className="text-slate-400 text-xs tracking-widest mb-1" />

                        <TextInput
                            type="password"
                            id="password_confirmation"
                            name="password_confirmation"
                            value={data.password_confirmation}
                            className="mt-1 block w-full bg-slate-950 border-slate-700 text-slate-200 focus:border-cyan-500 focus:ring-cyan-500/50 placeholder-slate-600 rounded-md"
                            autoComplete="new-password"
                            onChange={(e) => setData('password_confirmation', e.target.value)}
                            placeholder="••••••••"
                        />

                        <InputError message={errors.password_confirmation} className="mt-2 text-rose-400" />
                    </div>

                    {/* Tombol Aksi */}
                    <div className="mt-6 flex items-center justify-end">
                        <PrimaryButton 
                            className="ms-4 bg-cyan-600 hover:bg-cyan-500 text-white font-bold tracking-wider px-6 py-2 rounded-lg shadow-[0_0_10px_rgba(6,182,212,0.3)] transition-all border border-cyan-400" 
                            disabled={processing}
                        >
                            {processing ? 'UPDATING...' : 'RESET_PASSWORD'}
                        </PrimaryButton>
                    </div>
                </form>
            </div>

            {/* Footer */}
            <div className="mt-8 text-[10px] text-slate-600 tracking-widest">
                SECURE SYSTEM V1.0 • CREDENTIAL UPDATE
            </div>
        </div>
    );
}