#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optical propagation simulation with a cylindrical lens (1D FFT model).

Description
This script loads an SLM pattern from an 8-bit grayscale BMP and simulates the
focal plane of a *vertical* cylindrical lens (optical power along X). The lens
action is approximated as a 1D Fourier transform performed row-by-row along the
X axis (``axis=1``).

Two cases are shown:
- **PHASE**: the pattern modulates only the phase of the field (typical SLM case).
- **AMPLITUDE**: the pattern modulates the amplitude (idealized / polarizer-like case,
  useful for comparison).

Calibration note
In file naming, the convention ``w108`` is used to indicate that grayscale
level 108 corresponds to π radians. Here this scaling is controlled by ``VALORE_PI``.

Author: Fabrizio Coccetti
Date: 2026-01-23
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# --- CONFIGURAZIONE ---
# Assicurati che questo sia il file che contiene i dati (come visto da bmp_matrix_values.txt)
# FILENAME = "08_bits/propagation/SLM_CERN_event1221_480x480_r8_c8_w108.bmp"
# FILENAME = "08_bits/propagation/SLM_test_nbit8_480x480_r8_c8_w108.bmp"
# FILENAME = "08_bits/propagation/SLM_test_checkerboard_nbit8_480x480_r8_c8_w108.bmp"
# FILENAME = "08_bits/propagation/SLM_test_black_nbit8_480x480_r8_c8_w108.bmp"
# FILENAME = "08_bits/propagation/SLM_test_white_nbit8_480x480_r8_c8_w108.bmp"
# FILENAME = "08_bits/propagation/SLM_Pattern_Ready_217_Limit.bmp"
# FILENAME = "08_bits/propagation/SLM_Dark_Frame_217.bmp"
FILENAME = "08_bits/propagation/img/SLM_CERN_event1221_480x480_r8_c8_w108_grating.bmp"
# Se necessario, aggiusta il percorso:
IMG_PATH = os.path.join(os.getcwd(), FILENAME) 

# Parametri Fisici
# w108 nel nome suggerisce che il livello 108 corrisponde a Pi greco
VALORE_PI = 108.0 

def main():
    if not os.path.exists(IMG_PATH):
        print(f"ERRORE: File non trovato in {IMG_PATH}")
        # Cerchiamo nelle cartelle vicine per aiuto
        print("Cartella corrente:", os.getcwd())
        return

    # 1. Caricamento Immagine
    img = Image.open(IMG_PATH).convert('L')
    data = np.array(img, dtype=np.float32)
    
    print(f"Immagine Caricata. Max: {data.max()}, Min: {data.min()}")

    # 2. Preparazione dei Campi
    # CASO A: FASE PURA (Configurazione Standard SLM)
    # Background (0) -> Fase 0 -> Campo 1.0
    # Segnale (108) -> Fase Pi -> Campo -1.0
    # Nota: Normalizziamo a Pi greco basandoci sul nome del file
    phase_map = (data / VALORE_PI) * np.pi
    field_phase = 1.0 * np.exp(1j * phase_map)

    # CASO B: AMPIEZZA (Simulazione "Intuitiva" o con Polarizzatori)
    # Background (0) -> Opaco -> Campo 0.0
    # Segnale (>0) -> Trasparente -> Campo ~1.0
    # Questo serve a vedere "cosa succederebbe se bloccasisimo il fondo"
    field_amp = data / np.max(data) if np.max(data) > 0 else data

    # 3. Simulazione Lente Cilindrica Verticale
    # La lente verticale ha potere in X (orizzontale). Fa la FFT riga per riga (axis=1).
    # L'asse Y (verticale) rimane immagine.
    
    # FFT per la Fase
    fft_phase = np.fft.fftshift(np.fft.fft(field_phase, axis=1), axes=1)
    intensity_phase = np.abs(fft_phase)**2
    
    # FFT per l'Ampiezza
    fft_amp = np.fft.fftshift(np.fft.fft(field_amp, axis=1), axes=1)
    intensity_amp = np.abs(fft_amp)**2

    # 4. Plotting
    plt.figure(figsize=(15, 8))

    # Input
    plt.subplot(2, 3, 1)
    plt.title("Input Image (Segnale)")
    plt.imshow(data, cmap='gray', aspect='auto')
    plt.colorbar(label="Grigio (0-255)")

    # Risultato FASE (Log Scale per vedere dettagli)
    plt.subplot(2, 3, 2)
    plt.title("Simulazione FASE (Reale SLM)\nNota il picco centrale del fondo!")
    plt.imshow(np.log1p(intensity_phase), cmap='inferno', aspect='auto')
    plt.colorbar(label="Log Intensità")

    # Profilo FASE (Riga centrale)
    plt.subplot(2, 3, 3)
    plt.title("Profilo FASE (Media sulle righe)")
    plt.plot(np.mean(intensity_phase, axis=0))
    plt.xlabel("Posizione Focale X")
    plt.ylabel("Intensità Media")

    # Risultato AMPIEZZA
    plt.subplot(2, 3, 5)
    plt.title("Simulazione AMPIEZZA (Ideale/Polarizzata)\nSolo il segnale contribuisce")
    plt.imshow(np.log1p(intensity_amp), cmap='inferno', aspect='auto')
    plt.colorbar(label="Log Intensità")

    # Profilo AMPIEZZA
    plt.subplot(2, 3, 6)
    plt.title("Profilo AMPIEZZA (Media sulle righe)")
    plt.plot(np.mean(intensity_amp, axis=0))
    plt.xlabel("Posizione Focale X")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
