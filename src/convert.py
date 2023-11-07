import numpy as np
import pretty_midi
import tempfile
from scipy.stats import mode
from collections import Counter, OrderedDict, defaultdict
import subprocess
import os
import inspect
from scipy.io.wavfile import read as wavread, write as wavwrite
import distutils.spawn
from bintypes import *
from apu import *
import math

def exprsco_downsample(exprsco, rate, adaptive):
  rate_prev, nsamps, exprsco = exprsco
  assert abs(rate_prev - 44100.) < 1e-6
  if abs(rate - 44100.) < 1e-6:
    return (rate_prev, nsamps, exprsco)

  if rate is None:
    # Find note onsets
    ch_to_last_note = {ch:0 for ch in range(4)}
    ch_to_onsets = defaultdict(list)
    for i in range(exprsco.shape[0]):
      for j in range(4):
        last_note = ch_to_last_note[j]
        note = exprsco[i, j, 0]
        if note > 0 and note != last_note:
          ch_to_onsets[j].append(i)
        ch_to_last_note[j] = note

    # Find note intervals
    intervals = Counter()
    for _, onsets in ch_to_onsets.items():
      for i in range(1, len(onsets)):
        interval = onsets[i] - onsets[i - 1]
        # Minimum length 1ms
        if interval > 44:
          intervals[interval] += 1
    intervals_t = {i / 44100.:c for i, c in intervals.items()}

    # Raise error if we don't have enough info to estimate tempo
    num_intervals = sum(intervals.values())
    if num_intervals < 10:
      raise Exception('Too few intervals ({}) to estimate tempo'.format(num_intervals))

    # Determine how well each rate divides the intervals
    # TODO: Make 1/24/disc variables
    disc = 0.001
    rate_errors = []
    for rate in np.arange(1., 24. + disc, disc):
      error = 0.
      for interval, count in intervals_t.items():
        quotient = math.floor(interval * rate + 1e-8)
        remainder = interval - (quotient / rate)
        if remainder < 0 and abs(remainder) < 1e-8:
          remainder = 0
        assert remainder >= 0
        error += remainder * count
      rate_errors.append((rate, error))

    # Find best rate
    rate_errors = sorted(rate_errors, key=lambda x: x[1])
    rate = float(rate_errors[0][0])

  # Downsample
  rate = float(rate)
  ndown = int((nsamps / 44100.) * rate)
  score_low = np.zeros((ndown, 4, 3), dtype=np.uint8)
  for i in range(ndown):
    t_lo = i / rate
    t_hi = (i + 1) / rate
    # TODO: round these instead of casting?
    samp_lo, samp_hi = [int(t * 44100.) for t in [t_lo, t_hi]]
    score_slice = exprsco[samp_lo:samp_hi]
    for ch in range(4):
      score_slice_ch = score_slice[:, ch, :]
      on_frames = np.where(score_slice_ch[:, 0] != 0)[0]
      if len(on_frames) > 0:
        score_low[i, ch, :] = mode(score_slice_ch[on_frames])[0][0]
      else:
        score_low[i, ch, :2] = 0
        score_low[i, ch, 2] = mode(score_slice_ch[:, 2])[0][0]

  return (rate, nsamps, score_low)

def exprsco_to_rawsco(exprsco, clock=1789773.):
  rate, nsamps, exprsco = exprsco

  m = exprsco[:, :3, 0]
  m_zero = np.where(m == 0)

  m = m.astype(np.float32)
  f = 440 * np.power(2, ((m - 69) / 12))

  f_p, f_tr = f[:, :2], f[:, 2:]

  t_p = np.round((clock / (16 * f_p)) - 1)
  t_tr = np.round((clock / (32 * f_tr)) - 1)
  t = np.concatenate([t_p, t_tr], axis=1)

  t = t.astype(np.uint16)
  t[m_zero] = 0
  th = np.right_shift(np.bitwise_and(t, 0b11100000000), 8)
  tl = np.bitwise_and(t, 0b00011111111)

  rawsco = np.zeros((exprsco.shape[0], 4, 4), dtype=np.uint8)
  rawsco[:, :, 2:] = exprsco[:, :, 1:]
  rawsco[:, :3, 0] = th
  rawsco[:, :3, 1] = tl
  rawsco[:, 3, 1:] = exprsco[:, 3, :]

  return (clock, rate, nsamps, rawsco)

def exprsco_to_midi(exprsco):
  rate, nsamps, exprsco = exprsco

  # Create MIDI instruments
  p1_prog = pretty_midi.instrument_name_to_program('Lead 1 (square)')
  p2_prog = pretty_midi.instrument_name_to_program('Lead 2 (sawtooth)')
  tr_prog = pretty_midi.instrument_name_to_program('Synth Bass 1')
  no_prog = pretty_midi.instrument_name_to_program('Breath Noise')
  p1 = pretty_midi.Instrument(program=p1_prog, name='p1', is_drum=False)
  p2 = pretty_midi.Instrument(program=p2_prog, name='p2', is_drum=False)
  tr = pretty_midi.Instrument(program=tr_prog, name='tr', is_drum=False)
  no = pretty_midi.Instrument(program=no_prog, name='no', is_drum=True)

  # Iterate through score to extract channel notes
  notes = {}
  ccs = {}
  for i, ch in enumerate(np.split(exprsco, 4, axis=1)):
    ch = ch[:, 0, :]

    # MIDI doesn't allow velocity 0 messages so set tr velocity to 1
    if i == 2:
      ch[:, 1] = 1
      last_velocity = 1
    else:
      last_velocity = 0

    last_note = 0
    last_timbre = 0
    note_starts = []
    note_ends = []
    ch_ccs = []
    for s, (note, velocity, timbre) in enumerate(ch):
      if note != last_note:
        if note == 0:
          note_ends.append(s)
        else:
          if last_note == 0:
            note_starts.append((s, note, velocity))
          else:
            note_ends.append(s)
            note_starts.append((s, note, velocity))
      else:
        if velocity != last_velocity:
          ch_ccs.append((s, 11, velocity))

      if timbre != last_timbre:
        ch_ccs.append((s, 12, timbre))

      last_note = note
      last_velocity = velocity
      last_timbre = timbre
    if last_note != 0:
      note_ends.append(s + 1)

    assert len(note_starts) == len(note_ends)
    notes[i] = zip(note_starts, note_ends)
    ccs[i] = ch_ccs

  # Add notes to MIDI instruments
  for i, ins in enumerate([p1, p2, tr, no]):
    for (start_samp, note, velocity), end_samp in notes[i]:
      assert end_samp > start_samp
      start_t, end_t = start_samp / 44100., end_samp / 44100.
      note = pretty_midi.Note(velocity=velocity, pitch=note, start=start_t, end=end_t)
      ins.notes.append(note)

    for samp, cc_num, arg in ccs[i]:
      cc = pretty_midi.ControlChange(cc_num, arg, samp / 44100.)
      ins.control_changes.append(cc)

  # Add instruments to MIDI file
  midi = pretty_midi.PrettyMIDI(initial_tempo=120, resolution=22050)
  midi.instruments.extend([p1, p2, tr, no])

  # Create indicator for end of song
  eos = pretty_midi.TimeSignature(1, 1, nsamps / 44100.)
  midi.time_signature_changes.append(eos)

  # Write/read MIDI file
  mf = tempfile.NamedTemporaryFile('rb')
  midi.write(mf.name)
  midi = mf.read()
  mf.close()

  return midi

def midi_to_exprsco(midi):

  # Write/read MIDI file
  mf = tempfile.NamedTemporaryFile('wb')
  mf.write(midi)
  mf.seek(0)
  fp = mf.name
  
  print(fp)
  midi = pretty_midi.PrettyMIDI('./15_midi.mid')
  mf.close()
  print(midi)

  # Recover number of samples from time signature change indicator
  print(midi.time_signature_changes)
  assert len(midi.time_signature_changes) == 2
  nsamps = int(np.round(midi.time_signature_changes[1].time * 44100))

  # Find voices in MIDI
  exprsco = np.zeros((nsamps, 4, 3), dtype=np.uint8)
  ins_names = ['p1', 'p2', 'tr', 'no']
  for ins in midi.instruments:
    ch = ins_names.index(ins.name)

    # Process note messages
    comms = defaultdict(list)
    for note in ins.notes:
      start = int(np.round(note.start * 44100))
      end = int(np.round(note.end * 44100))
      velocity = note.velocity if ch != 2 else 0
      note = note.pitch

      comms[start].append(('note_on', note, velocity))
      comms[end].append(('note_off',))

    # Process CC messages
    for cc in ins.control_changes:
      samp = int(np.round(cc.time * 44100))
      if cc.number == 11:
        velocity = cc.value
        assert velocity > 0
        comms[samp].append(('cc_velo', velocity))
      elif cc.number == 12:
        timbre = cc.value
        comms[samp].append(('cc_timbre', timbre))
      else:
        assert False

    # Write score
    note = 0
    velocity = 0
    timbre = 0
    for i in range(nsamps):
      for comm in comms[i]:
        if comm[0] == 'note_on':
          note = comm[1]
          velocity = comm[2]
        elif comm[0] == 'note_off':
          note = 0
          velocity = 0
        elif comm[0] == 'cc_velo':
          velocity = comm[1]
        elif comm[0] == 'cc_timbre':
          timbre = comm[1]
        else:
          assert False

      exprsco[i, ch] = (note, velocity, timbre)

  return 44100, nsamps, exprsco

def rawsco_to_ndf(rawsco):
  fs = 44100.

  clock, rate, nsamps, score = rawsco

  if rate == 44100:
    ar = True
  else:
    ar = False

  max_i = score.shape[0]

  samp = 0
  t = 0.
  # ('apu', ch, func, func_val, natoms, offset)
  ndf = [
      ('clock', int(clock)),
      ('apu', 'ch', 'p1', 0, 0, 0),
      ('apu', 'ch', 'p2', 0, 0, 0),
      ('apu', 'ch', 'tr', 0, 0, 0),
      ('apu', 'ch', 'no', 0, 0, 0),
      ('apu', 'p1', 'du', 0, 1, 0),
      ('apu', 'p1', 'lh', 1, 1, 0),
      ('apu', 'p1', 'cv', 1, 1, 0),
      ('apu', 'p1', 'vo', 0, 1, 0),
      ('apu', 'p1', 'ss', 7, 2, 1), # This is necessary to prevent channel silence for low notes
      ('apu', 'p2', 'du', 0, 3, 0),
      ('apu', 'p2', 'lh', 1, 3, 0),
      ('apu', 'p2', 'cv', 1, 3, 0),
      ('apu', 'p2', 'vo', 0, 3, 0),
      ('apu', 'p2', 'ss', 7, 4, 1), # This is necessary to prevent channel silence for low notes
      ('apu', 'tr', 'lh', 1, 5, 0),
      ('apu', 'tr', 'lr', 127, 5, 0),
      ('apu', 'no', 'lh', 1, 6, 0),
      ('apu', 'no', 'cv', 1, 6, 0),
      ('apu', 'no', 'vo', 0, 6, 0),
  ]
  ch_to_last_tl = {ch:0 for ch in ['p1', 'p2']}
  ch_to_last_th = {ch:0 for ch in ['p1', 'p2']}
  ch_to_last_timer = {ch:0 for ch in ['p1', 'p2', 'tr']}
  ch_to_last_du = {ch:0 for ch in ['p1', 'p2']}
  ch_to_last_volume = {ch:0 for ch in ['p1', 'p2', 'no']}
  last_no_np = 0
  last_no_nl = 0

  for i in range(max_i):
    for j, ch in enumerate(['p1', 'p2']):
      th, tl, volume, du = score[i, j]
      timer = (th << 8) + tl
      last_timer = ch_to_last_timer[ch]

      # NOTE: This will never be perfect reconstruction because phase is not incremented when the channel is off
      retrigger = False
      if last_timer == 0 and timer != 0:
        ndf.append(('apu', 'ch', ch, 1, 0, 0))
        retrigger = True
      elif last_timer != 0 and timer == 0:
        ndf.append(('apu', 'ch', ch, 0, 0, 0))

      if du != ch_to_last_du[ch]:
        ndf.append(('apu', ch, 'du', du, 0, 0))
        ch_to_last_du[ch] = du

      if volume > 0 and volume != ch_to_last_volume[ch]:
        ndf.append(('apu', ch, 'vo', volume, 0, 0))
      ch_to_last_volume[ch] = volume

      if tl != ch_to_last_tl[ch]:
        ndf.append(('apu', ch, 'tl', tl, 0, 2))
        ch_to_last_tl[ch] = tl
      if retrigger or th != ch_to_last_th[ch]:
        ndf.append(('apu', ch, 'th', th, 0, 3))
        ch_to_last_th[ch] = th

      ch_to_last_timer[ch] = timer

    j = 2
    ch = 'tr'
    th, tl, _, _ = score[i, j]
    timer = (th << 8) + tl
    last_timer = ch_to_last_timer[ch]
    if last_timer == 0 and timer != 0:
      ndf.append(('apu', 'ch', ch, 1, 0, 0))
    elif last_timer != 0 and timer == 0:
      ndf.append(('apu', 'ch', ch, 0, 0, 0))
    if timer != last_timer:
      ndf.append(('apu', ch, 'tl', tl, 0, 2))
      ndf.append(('apu', ch, 'th', th, 0, 3))
    ch_to_last_timer[ch] = timer

    j = 3
    ch = 'no'
    _, np, volume, nl = score[i, j]
    if last_no_np == 0 and np != 0:
      ndf.append(('apu', 'ch', ch, 1, 0, 0))
    elif last_no_np != 0 and np == 0:
      ndf.append(('apu', 'ch', ch, 0, 0, 0))
    if volume > 0 and volume != ch_to_last_volume[ch]:
      ndf.append(('apu', ch, 'vo', volume, 0, 0))
    ch_to_last_volume[ch] = volume
    if nl != last_no_nl:
      ndf.append(('apu', ch, 'nl', nl, 0, 2))
      last_no_nl = nl
    if np > 0 and np != last_no_np:
      ndf.append(('apu', ch, 'np', 16 - np, 0, 2))
      ndf.append(('apu', ch, 'll', 0, 0, 3))
    last_no_np = np

    if ar:
      wait_amt = 1
    else:
      t += 1. / rate
      wait_amt = min(int(fs * t) - samp, nsamps - samp)

    ndf.append(('wait', wait_amt))
    samp += wait_amt

  remaining = nsamps - samp
  assert remaining >= 0
  if remaining > 0:
    ndf.append(('wait', remaining))

  return ndf

def ndf_to_ndr(ndf):
  ndr = ndf[:1]
  ndf = ndf[1:]

  registers = {
      'p1': [0x00] * 4,
      'p2': [0x00] * 4,
      'tr': [0x00] * 4,
      'no': [0x00] * 4,
      'dm': [0x00] * 4,
      'ch': [0x00],
      'fc': [0x00]
  }

  # Convert commands to VGM
  regn_to_val = OrderedDict()
  for comm in ndf:
    itype = comm[0]
    if itype == 'wait':
      for _, (arg1, arg2) in regn_to_val.items():
        ndr.append(('apu', b2h(c2b(arg1)), b2h(c2b(arg2))))
      regn_to_val = OrderedDict()

      amt = comm[1]

      ndr.append(('wait', amt))
    elif itype == 'apu':
      dest = comm[1]
      param = comm[2]
      val = comm[3]
      natoms = comm[4]
      param_offset = comm[5]

      # Find offset/bitmask
      reg = registers[dest]
      param_bitmask = func_to_bitmask(dest, param)

      # Apply mask
      mask_bin = '{:08b}'.format(param_bitmask)
      nbits = mask_bin.count('1')
      if val < 0 or val >= (2 ** nbits):
        raise ValueError('{}, {} (0, {}]: invalid value specified {}'.format(comm[1], comm[2], (2 ** nbits), val))
      assert val >= 0 and val < (2 ** nbits)
      shift = max(0, 7 - mask_bin.rfind('1')) % 8
      val_old = reg[param_offset]
      reg[param_offset] &= (255 - param_bitmask)
      reg[param_offset] |= val << shift
      assert reg[param_offset] < 256
      val_new = reg[param_offset]

      arg1 = register_memory_offsets[dest] + param_offset
      arg2 = reg[param_offset]

      regn_to_val[(dest, param_offset, natoms)] = (arg1, arg2)
    elif itype == 'ram':
      # TODO
      continue
    else:
      raise NotImplementedError()

  for _, (arg1, arg2) in regn_to_val.items():
    ndr.append(('apu', b2h(c2b(arg1)), b2h(c2b(arg2))))

  return ndr

def ndr_to_vgm(ndr):
  assert ndr[0][0] == 'clock'
  clock = ndr[0][1]

  ndr = ndr[1:]

  EMPTYBYTE = i2lub(0)
  flatten = lambda vgm: list(''.join(vgm))
  vgm = flatten([EMPTYBYTE] * 48)

  # VGM identifier
  vgm[:0x04] = [c2b(c) for c in [0x56, 0x67, 0x6d, 0x20]]
  # Version
  vgm[0x08:0x0c] = i2lub(0x161)
  # Clock rate
  vgm[0x84:0x88] = i2lub(clock)
  # Data offset
  vgm[0x34:0x38] = i2lub(0xc0 - 0x34)

  wait_sum = 0
  for comm in ndr:
    itype = comm[0]
    if itype == 'wait':
      amt = comm[1]
      wait_sum += amt

      while amt > 65535:
        vgm.append(c2b(0x61))
        vgm.append(i2lusb(65535))
        amt -= 65535

      vgm.append(c2b(0x61))
      vgm.append(i2lusb(amt))
    elif itype == 'apu':
      arg1 = h2b(comm[1])
      arg2 = h2b(comm[2])
      vgm.append(c2b(0xb4))
      vgm.append(arg1)
      vgm.append(arg2)
    elif itype == 'ram':
      raise NotImplementedError()
    else:
      raise NotImplementedError()

  # Halt
  vgm.append(c2b(0x66))
  vgm = flatten(vgm)

  # Total samples
  vgm[0x18:0x1c] = i2lub(wait_sum)
  # EoF offset
  vgm[0x04:0x08] = i2lub(len(vgm) - 0x04)

  vgm = ''.join(vgm)
  return vgm

def load_vgmwav(wav_fp):
  fs, wav = wavread(wav_fp)
  assert fs == 44100
  if wav.ndim == 2:
    wav = wav[:, 0]
  wav = wav.astype(np.float32)
  wav /= 32767.
  return wav

def vgm_to_wav(vgm):
  # Try to get binary fp from build dir
  bin_fp = None
  try:
    bin_dir = os.path.dirname()
    bin_fp = os.path.join(bin_dir, 'vgm2wav')
  except:
    pass

  # Try to get binary fp at ${VGMTOWAV}
  try:
    env_var = os.environ['./']
    bin_fp = env_var
  except:
    pass

  # Make sure it is accessible
  if bin_fp is not None:
    if not (os.path.isfile(bin_fp) and os.access(bin_fp, os.X_OK)):
      raise Exception('vgm2wav should be at \'{}\' but it does not exist or is not executable'.format(bin_fp))

  # Try finding it on global path otherwise
  if bin_fp is None:
    bin_fp = distutils.spawn.find_executable('vgm2wav')

  # Ensure vgm2wav was found
  if bin_fp is None:
    raise Exception('Could not find vgm2wav executable. Please set $VGMTOWAV environment variable')

  vf = tempfile.NamedTemporaryFile('wb')
  wf = tempfile.NamedTemporaryFile('rb')

  vf.write(vgm)
  vf.seek(0)

  res = subprocess.call('{} --loop-count 1 {} {}'.format(bin_fp, vf.name, wf.name).split())
  if res > 0:
    vf.close()
    wf.close()
    raise Exception('Invalid return code {} from vgm2wav'.format(res))

  vf.close()

  wf.seek(0)
  wav = load_vgmwav(wf.name)

  wf.close()

  return wav

def save_vgmwav(wav_fp, wav):
  wav *= 32767.
  wav = np.clip(wav, -32768., 32767.)
  wav = wav.astype(np.int16)
  wavwrite(wav_fp, 44100, wav)

def midi_to_wav(midi, midi_to_wav_rate=None):
  exprsco = midi_to_exprsco(midi)
  if midi_to_wav_rate is not None:
    exprsco = exprsco_downsample(exprsco, midi_to_wav_rate, False)
  rawsco = exprsco_to_rawsco(exprsco)
  ndf = rawsco_to_ndf(rawsco)
  ndr = ndf_to_ndr(ndf)
  vgm = ndr_to_vgm(ndr)
  wav = vgm_to_wav(vgm)
  return wav

with open('15_midi.mid', 'rb') as f:
  mid = f.read()
print('testando...')
wav = midi_to_wav(mid, midi_to_wav_rate=100)

save_vgmwav('/', wav)
print('sucesso')