# Kit deep probe (runs after the Models and Setup cells; no ComfyUI server needed). KIT_MODEL_FILE, KIT_TOKENS come from the builder.
fork = COMFY / 'custom_nodes' / 'ComfyUI-MultiGPU' / 'portkit' / 'probe_model.py'
model = next(M.glob('diffusion_models/' + KIT_MODEL_FILE))
r = subprocess.run([sys.executable, '-I', str(fork), '--model', str(model), '--out', str(OUT / 'probe'), '--deep', '--comfy', str(COMFY), '--tokens', str(KIT_TOKENS)],
                   capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES='0,1'))
say('probe rc %d' % r.returncode)
say((r.stdout + r.stderr)[-6000:])
