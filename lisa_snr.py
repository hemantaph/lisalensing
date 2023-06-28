import numpy as np
import contextlib
import json
from tqdm import tqdm
from multiprocessing import Pool

import lisabeta.lisa.lisa as lisa

class LISA_SNR:

    def __init__(self, npool=int(4), waveform_params=None):
        self.npool = npool
        if waveform_params:
            self.waveform_params = waveform_params
        else:
            self.waveform_params = {
                # Frequency range
                "minf": 1e-5,
                "maxf": 0.5,
                # Reference epoch of coalescence, yr -- coalescence is at t0*yr + Deltat*s, Deltat in params
                "t0": 0.0,
                # Always cut signals timetomerger_max*yr before merger -- to avoid needlessly long signals using minf
                "timetomerger_max": 1.0,
                # Option to cut the signal pre-merger -- must be in L-frame
                "DeltatL_cut": None,
                # Further options to cut signals
                "fend": None,
                "tmin": None,
                "tmax": None,
                # Options for the time and phase alignment -- development/testing
                "phiref": 0.0,
                "fref_for_phiref": 0.0,
                "tref": 0.0,
                "fref_for_tref": 0.0,
                "force_phiref_fref": True,
                "toffset": 0.0,
                # TDI channels to generate
                "TDI": "TDIAET",
                # Internal accuracy params
                "acc": 1e-4,
                "order_fresnel_stencil": 0,
                # Waveform approximant and set of harmonics to use
                "approximant": "IMRPhenomHM",
                "modes": None,
                # LISA response options
                "LISAconst": "Proposal",
                "responseapprox": "full",
                "frozenLISA": False,
                "TDIrescaled": True,
                # Noise options -- can also be given as a numpy array for interpolation
                "LISAnoise": {
                    "InstrumentalNoise": "SciRDv1",
                    "WDbackground": False,
                    "WDduration" : 0.0,
                    "lowf_add_pm_noise_f0": 0.0,
                    "lowf_add_pm_noise_alpha": 2.0
                }
            }


    # calculate the snr with multiprocessing 
    def snr_mp(
        self,
        mass_1=[3e6, 4e6],
        mass_2=[1e6, 1e6],
        luminosity_distance = 3.65943e+04,
        iota = 1.0471975511965976,
        psi = 1.7,
        phase = 1.2,
        Deltat = 0.0,
        longitude = 0.8,
        latitude = 0.3,
        a_1 = 0.5,
        a_2 = 0.2,
        GWparam_dict=False,
        verbose=True,
        jsonFile=False,
    ):

        if GWparam_dict != False:
            mass_1 = GWparam_dict["mass_1"]
            mass_2 = GWparam_dict["mass_2"]
            luminosity_distance = GWparam_dict["luminosity_distance"]
            iota = GWparam_dict["iota"]
            psi = GWparam_dict["psi"]
            phase = GWparam_dict["phase"]
            Deltat = GWparam_dict["Deltat"]
            longitude = GWparam_dict["longitude"]
            latitude = GWparam_dict["latitude"]
            a_1 = GWparam_dict["a_1"]
            a_2 = GWparam_dict["a_2"]

        
        # reshaping params
        mass_1, mass_2 = np.array([mass_1]).reshape(-1), np.array([mass_2]).reshape(-1)
        size = len(mass_1)
        luminosity_distance = np.array([luminosity_distance]).reshape(-1) * np.ones(size)
        iota = np.array([iota]).reshape(-1) * np.ones(size)
        psi = np.array([psi]).reshape(-1) * np.ones(size)
        phase = np.array([phase]).reshape(-1) * np.ones(size)
        Deltat = np.array([Deltat]).reshape(-1) * np.ones(size)
        longitude = np.array([longitude]).reshape(-1) * np.ones(size)
        latitude = np.array([latitude]).reshape(-1) * np.ones(size)
        a_1 = np.array([a_1]).reshape(-1) * np.ones(size)
        a_2 = np.array([a_2]).reshape(-1) * np.ones(size)

        Mtot = mass_1 + mass_2
        # mass_1 should be the larger mass
        q = mass_1 / mass_2
        
        params = np.zeros(size, dtype=object)
        for i in range(size):
            params[i] = {
                # Total *redshifted* mass M=m1+m2, solar masses
                "M": Mtot[i],
                # Mass ratio q=m1/m2
                "q": q[i],
                # Dimensionless spin component 1 along orbital momentum
                "chi1": a_1[i],
                # Dimensionless spin component 2 along orbital momentum
                "chi2": a_2[i],
                # Time shift of coalescence, s -- coalescence is at t0*yr + Deltat*s, t0 in waveform_params
                "Deltat": Deltat[i],
                # Luminosity distance, Mpc
                "dist": luminosity_distance[i],
                # Inclination, observer's colatitude in source-frame
                "inc": iota[i],
                # Phase, observer's longitude in source-frame
                "phi": phase[i],
                # Longitude in the sky
                "lambda": longitude[i],
                # Latitude in the sky
                "beta": latitude[i],
                # Polarization angle
                "psi": psi[i],
                # Flag indicating whether angles and Deltat pertain to the L-frame or SSB-frame
                "Lframe": True
            }

        params = np.array([params]).T

        waveform_params = self.waveform_params.copy()  
        waveform_params = np.array([np.full(size, waveform_params, dtype=object)]).T  

        iterations = np.arange(size).reshape(size,1)
        # concatinating the params, waveform_params and iterations
        input_arguments = np.concatenate((params, waveform_params, iterations), axis=1)

        iter_ = []
        SNRs_list = []
        # multiprocessing
        with contextlib.redirect_stdout(None):
            with Pool(processes=self.npool) as pool:
                # call the same function with different data in parallel
                # imap->retain order in the list, while map->doesn't
                for result in tqdm(
                    pool.imap_unordered(
                        self.lisabeta_snr, input_arguments
                    ),
                    total=size,
                    ncols=100,
                    disable=not verbose,
                ):
                    iter_.append(result[1])
                    SNRs_list.append(result[0])

        # to fill in the snr values at the right index
        snrs_ = np.zeros(size)
        snrs_[iter_] = SNRs_list[:]

        # make a dictionary of the input params and snr values
        snr_dict = {
            "mass_1": list(mass_1),
            "mass_2": list(mass_2),
            "luminosity_distance": list(luminosity_distance),
            "iota": list(iota),
            "psi": list(psi),
            "phase": list(phase),
            "Deltat": list(Deltat),
            "longitude": list(longitude),
            "latitude": list(latitude),
            "a_1": list(a_1),
            "a_2": list(a_2),
            "snr": list(snrs_),
        }

        # save the snr values as a json file
        if jsonFile != False:
            with open(jsonFile, "w") as fp:
                json.dump(snr_dict, fp)

        self.param_dict = snr_dict
        return snrs_
    
    # function to calculate the snr for a single set of params
    # multiprocessing requires a function with a single argument
    def lisabeta_snr(self, input):

        params = input[0]
        waveform_params = input[1]
        i = input[2]

        try:
            tdisignal = lisa.GenerateLISATDISignal_SMBH(params, **waveform_params)
            snr = tdisignal['SNR']
        except:
            snr = 0.0
        
        return snr, i
    

    # calculate the snr with multiprocessing 
    def snr(
        self,
        mass_1=[3e6, 4e6],
        mass_2=[1e6, 1e6],
        luminosity_distance = 3.65943e+04,
        iota = 1.0471975511965976,
        psi = 1.7,
        phase = 1.2,
        Deltat = 0.0,
        longitude = 0.8,
        latitude = 0.3,
        a_1 = 0.5,
        a_2 = 0.2,
        GWparam_dict=False,
        verbose=True,
        jsonFile=False,
    ):

        if GWparam_dict != False:
            mass_1 = GWparam_dict["mass_1"]
            mass_2 = GWparam_dict["mass_2"]
            luminosity_distance = GWparam_dict["luminosity_distance"]
            iota = GWparam_dict["iota"]
            psi = GWparam_dict["psi"]
            phase = GWparam_dict["phase"]
            Deltat = GWparam_dict["Deltat"]
            longitude = GWparam_dict["longitude"]
            latitude = GWparam_dict["latitude"]
            a_1 = GWparam_dict["a_1"]
            a_2 = GWparam_dict["a_2"]

        
        # reshaping params
        mass_1, mass_2 = np.array([mass_1]).reshape(-1), np.array([mass_2]).reshape(-1)
        size = len(mass_1)
        luminosity_distance = np.array([luminosity_distance]).reshape(-1) * np.ones(size)
        iota = np.array([iota]).reshape(-1) * np.ones(size)
        psi = np.array([psi]).reshape(-1) * np.ones(size)
        phase = np.array([phase]).reshape(-1) * np.ones(size)
        Deltat = np.array([Deltat]).reshape(-1) * np.ones(size)
        longitude = np.array([longitude]).reshape(-1) * np.ones(size)
        latitude = np.array([latitude]).reshape(-1) * np.ones(size)
        a_1 = np.array([a_1]).reshape(-1) * np.ones(size)
        a_2 = np.array([a_2]).reshape(-1) * np.ones(size)

        Mtot = mass_1 + mass_2
        # mass_1 should be the larger mass
        q = mass_1 / mass_2

        waveform_params = self.waveform_params.copy()
        snr = np.zeros(size)
        with contextlib.redirect_stdout(None):
            for i in range(size):

                params = {
                    # Total *redshifted* mass M=m1+m2, solar masses
                    "M": Mtot[i],
                    # Mass ratio q=m1/m2
                    "q": q[i],
                    # Dimensionless spin component 1 along orbital momentum
                    "chi1": a_1[i],
                    # Dimensionless spin component 2 along orbital momentum
                    "chi2": a_2[i],
                    # Time shift of coalescence, s -- coalescence is at t0*yr + Deltat*s, t0 in waveform_params
                    "Deltat": Deltat[i],
                    # Luminosity distance, Mpc
                    "dist": luminosity_distance[i],
                    # Inclination, observer's colatitude in source-frame
                    "inc": iota[i],
                    # Phase, observer's longitude in source-frame
                    "phi": phase[i],
                    # Longitude in the sky
                    "lambda": longitude[i],
                    # Latitude in the sky
                    "beta": latitude[i],
                    # Polarization angle
                    "psi": psi[i],
                    # Flag indicating whether angles and Deltat pertain to the L-frame or SSB-frame
                    "Lframe": True
                }

                # SNR
                try:
                    tdisignal = lisa.GenerateLISATDISignal_SMBH(params, **waveform_params)
                    snr[i] = tdisignal['SNR']
                    
                except:
                    #print("SNR calculation failed")
                    snr[i] = 0.0

        return snr
    

# calculate the snr for lensed events with multiprocessing
def snr_lensed(lisa_snr_class, lensed_param, n_max_images=4):
    """
    Function to calculate the optimal signal to noise ratio for lensed events

    Parameters
    ----------
    lensed_param : dict
        Dictionary containing the lensed parameters
    n_max_images : int
        Maximum number of images to consider in an event
    """
    magnifications = lensed_param["magnifications"]
    time_delays = lensed_param["time_delays"]

    # Get the binary parameters
    number_of_lensed_events = len(magnifications)
    mass_1, mass_2, luminosity_distance, iota, psi, phase, Deltat, longitude, latitude, a_1, a_2  = (
        lensed_param["mass_1"],
        lensed_param["mass_2"],
        lensed_param["luminosity_distance"],
        lensed_param["iota"],
        lensed_param["psi"],
        lensed_param["phase"],
        lensed_param["Deltat"],
        lensed_param["longitude"],
        lensed_param["latitude"],
        lensed_param["a_1"],
        lensed_param["a_2"],
    )

    # setting up snr dictionary
    optimal_snrs = dict()
    optimal_snrs["opt_snr_net"] = (
        np.ones((number_of_lensed_events, n_max_images)) * np.nan
    )

    for i in range(n_max_images):
        # Get the optimal signal to noise ratios for each image
        buffer = magnifications[:, i]
        idx = ~np.isnan(buffer)  # index of not-nan
        effective_luminosity_distance = luminosity_distance[idx] / np.sqrt(
            np.abs(buffer[idx])
        )
        effective_Deltat = Deltat[idx] + time_delays[idx, i]
        # Each image has their own effective luminosity distance
        if len(effective_luminosity_distance) != 0:
            # Returns a dictionary
            lisa_ = lisa_snr_class()
            optimal_snr = lisa_.snr_mp(
                mass_1 = mass_1[idx],
                mass_2 = mass_2[idx],
                luminosity_distance = effective_luminosity_distance,
                iota = iota[idx],
                psi = psi[idx],
                phase = phase[idx],
                Deltat = effective_Deltat,
                longitude = longitude[idx],
                latitude = latitude[idx],
                a_1 = a_1[idx],
                a_2 = a_2[idx],
                jsonFile=False,
            )

            optimal_snrs["opt_snr_net"][idx, i] = optimal_snr

    return optimal_snrs

