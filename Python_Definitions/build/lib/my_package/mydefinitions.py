# importing libraries
import glob
import inspect
import math
import matplotlib.pyplot as plt
import numpy as np
from numpy import trapz
import pandas as pd
import plotly.express as px
from scipy.integrate import simpson
from scipy.integrate import simps
from scipy.optimize import curve_fit


def subtract_dark(data, dark_data = None):
    """ 
    Assumes 3D data (store to ram option on spectrometer), as it assumes [3:] columns are actual samples.
    Returns data in the same format (wavelength; dark; ...) but with dark current subtracted from samples.
    If dark_data is not specified, assumes column with index 1 in data to be dark current (true for 3D data)
    dark_data should be a one dimensional numpy array
    """

    data_new = np.zeros(data.shape)
    data_new[0:3] = data[0:3]

    if dark_data is not None:
        data_new[3:] = data[3:] - dark_data

    else:
        dark = data[1]
        data_new[3:] = data[3:] - dark

    

    return data_new


def plot_profile_values_changes(data):
    """
    Plots values of 4 pixels across measurements:
    takes pixel which has initially maximum value and plots its value acros measurements, aswell as 2 pixels before it and one pixel after it.

    Assumes 3D data
    """

    first_spectrum = data[3]
    max_index = np.argmax(first_spectrum)

    index_value = max_index 

    maxima_list_new = []
    for row_index in [num for num in range(3, len(data)  )]:
        maxima_list_new.append( data[row_index][index_value - 2 : index_value + 2] ) 

    # maxima_list_new is in form of lots of row where each row has 4 elements, we want to reshape it to 4 rows each for each pixel, hence transpose
    maxima_array_new = np.array(maxima_list_new).T


    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # First subplot in the top row
    axes[0, 0].plot(maxima_array_new[0], '.-')
    axes[0, 0].set_title('-2 from max')

    # Second subplot in the top row
    axes[0, 1].plot(maxima_array_new[1], '.-')
    axes[0, 1].set_title('-1 from max')


    axes[1, 0].plot(maxima_array_new[2], '.-')
    axes[1, 0].set_title('max pixel')


    axes[1, 1].plot(maxima_array_new[3], '.-')
    axes[1, 1].set_title('+1 from max')


    plt.tight_layout()
    plt.show()

    return maxima_array_new



def plot_two_profiles(data, index_0, index_1, wavelength_1):
    """ 
    Plots two spectra with small range on wavelength axis so that profile of peak can be seen.
    Two spectra area at index_0 and index_1, the wavelength around which plot is centered is wavelength_1

    Assumes 3D data (otherwise two profiles would not be in data)
    """

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    #index_0 = 4
    axes[0].plot(data[0], data[index_0], '.-')

    axes[0].set_title(f'Spectrum at index {index_0}')
    axes[0].set_xlabel('$\lambda$, $nm$')
    axes[0].set_ylabel('Counts')
    axes[0].set_xlim(wavelength_1 - 2 , wavelength_1 + 2)
    axes[0].grid(True)

    #index_1 = 805
    axes[1].plot(data[0], data[index_1], '.-')

    axes[1].set_title(f'Spectrum at index {index_1}')
    axes[1].set_xlabel('$\lambda$, $nm$')
    axes[1].set_ylabel('Counts')
    axes[1].set_xlim(wavelength_1 - 2 , wavelength_1 + 2)
    axes[1].grid(True)


    plt.tight_layout()


    plt.show()




def plot_maximum_counts(maxima_list):
    """ 
    Plots maximum value in each spectrum for all measurements.

    Assumes 3D data
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # First subplot
    axes[0].plot(maxima_list, '.-')
    axes[0].set_title('maximum counts')

    # Second subplot
    axes[1].plot(np.array(maxima_list) / maxima_list[0], '.-')
    #axes[1].set_xlim(0, 100)
    axes[1].set_title('normalized, maximum counts')

    plt.tight_layout()
    plt.show()



def plot_areas_under_spectra(data, normalize=True, plot_uncertainties=False, xlim_range=None, uncertainty_value=None, peak_only=False):
    """ 
    Assumes 3D data
    """
    areas_list = []
    areas_list_simpsons = []

    # for plotting only area under the peak
    if peak_only:
        first_spectrum = data[3]
        max_index = np.argmax(first_spectrum)

        for index in range(3, len(data)):
            
            areas_list.append( np.trapz(data[index][max_index-10 : max_index+10]) )
            area = simps( data[index][max_index-10 : max_index+10] )
            areas_list_simpsons.append(area)       

    else:
        for index in range(3, len(data)):
            areas_list.append(np.trapz(data[index]))
            area = simps(data[index])
            areas_list_simpsons.append(area)

    areas_array = np.array(areas_list)
    areas_array_simpsons = np.array(areas_list_simpsons)

    # update uncertainty if uncertainty value given
    if uncertainty_value is not None:
        uncertainty = uncertainty_value
    else: 
        uncertainty = 500

    # plotting
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    if normalize:
        axes[0].plot(areas_list / areas_list[0], '.-')
        axes[0].set_title('normalized areas under spectra, trapezoid')
        axes[0].set_xlabel('measurement number')

        axes[1].plot(areas_list_simpsons / areas_list_simpsons[0], '.-')
        axes[1].set_title('normalized, simpsons')

        if plot_uncertainties:
            axes[0].fill_between(range(len(areas_list)),
                                 (areas_array - uncertainty) / areas_array[0],
                                 (areas_array + uncertainty) / areas_array[0],
                                 alpha=0.3)

            axes[1].fill_between(range(len(areas_list_simpsons)),
                                 (areas_array_simpsons - uncertainty) / areas_list_simpsons[0],
                                 (areas_array_simpsons + uncertainty) / areas_list_simpsons[0],
                                 alpha=0.3)
    else:
        axes[0].plot(areas_list, '.-')
        axes[0].set_title('areas under spectra, trapezoid')
        axes[0].set_xlabel('measurement number')

        axes[1].plot(areas_list_simpsons, '.-')
        axes[1].set_title('simpsons')

        if plot_uncertainties:
            axes[0].fill_between(range(len(areas_list)),
                                 areas_array - uncertainty,
                                 areas_array + uncertainty,
                                 alpha=0.5)

            axes[1].fill_between(range(len(areas_list_simpsons)),
                                 areas_array_simpsons - uncertainty,
                                 areas_array_simpsons + uncertainty,
                                 alpha=0.5)

    # update xlim if xlim value given
    if xlim_range is not None:
        axes[1].set_xlim(xlim_range)

    plt.tight_layout()
    plt.show()

    return areas_list




def do_fft(data, time_interval_between_points, data_title):
    processed = (data - np.mean(data)) / np.var(data) ** (1/2)

    spectrum = np.abs(np.fft.fft(processed))[:len(processed)//2]
    frequencies = np.fft.fftfreq(len(processed), d=time_interval_between_points)[:len(processed)//2]

    fig = px.line(x=frequencies, y=spectrum)

    fig.update_layout(title=f'Fourier Transform {data_title}',
                      xaxis_title='Frequency [Hz]')

    fig.show()



    ################################################################################################################################
    # Part 2

def read_data(search_phrase, part_to_remove1, part_to_remove2, folder_path, skip_rows = None):
    """
    Takes files from folder_path, disregards files withouth search_phrase in their path, files that are considered have data imported from them.
    Indices are extracted for each file from the file name, by removing appropriate parts of path: part_to_remove 1 and part_to_remove2. 
    Entire data is then saved as table using pandas.DataFrame. The indices extracted from file name are indices in table (leftmost column), columns.values (top row) in
    the table is a list of wavelenghts (same for all data files), the elements of DataFrame are measured counts for appropriate spectra. 

    Input: 
    - search_phrase - string
    - part_to_remove 1, part_to_remove2 - sting
    - folder_path - string

    Output: 
    - all_data - DataFrame
    """
    
    all_spectra_list = [] # empty list to store spectra, where one spectrum is a list of values at different wavelengths
    indices_for_files_list = [] # for storing index at which given spectrum was obtained (for example angles or xy)

    # loop over txt files in the path
    for file in glob.glob(folder_path):

        # only files with search_phrase in name are considered, for example readme file is omitted as it contains no data
        if search_phrase in file:

            # replace strings defined above with empty space, effectively deleting them, then turn result into float
            index_for_file = float(file.replace(part_to_remove1, '').replace(part_to_remove2, ''))
            print(file) # print files that are considered
            print(index_for_file) # index associated with given file

            if skip_rows is not None:
                skip_rows_1 = skip_rows
            else:
                skip_rows_1 = 8
                
            # import data from the file, first 10 rows are strings and they are skipped
            data_from_file = np.loadtxt(file, delimiter = ';', skiprows = skip_rows_1, unpack = True)

            # add spectra to the list of spectra
            all_spectra_list.append(data_from_file[4])
            #all_spectra_list.append(np.sum(data_from_file[3:], axis = 0)/len(data_from_file[3:]))

            # just an x axis for plotting spectrum, array of wavelengths
            wavelengths_array = data_from_file[0]

            # update list of indices by number of degrees at which current spectrum was taken
            indices_for_files_list.append(index_for_file)

    # changing list to numpy array        
    indices_for_files_array = np.array(indices_for_files_list)

    # saving data in table format
    all_data = pd.DataFrame(data=all_spectra_list, index=indices_for_files_array, columns=wavelengths_array)

    all_data = all_data.sort_index()
    
    return all_data


def read_powermeter_csv(search_phrase, part_to_remove1, part_to_remove2, folder_path, skip_rows = None):

    
    powermeter_readings = [] 
    indices_for_files_list = [] # for storing index at which given spectrum was obtained (for example angles or xy)

    # loop over txt files in the path
    for file in glob.glob(folder_path):

        # only files with search_phrase in name are considered, for example readme file is omitted as it contains no data
        if search_phrase in file:

            # replace strings defined above with empty space, effectively deleting them, then turn result into float
            index_for_file = float(file.replace(part_to_remove1, '').replace(part_to_remove2, ''))
            print(file) # print files that are considered
            print(index_for_file) # index associated with given file

            if skip_rows is not None:
                skip_rows_1 = skip_rows
            else:
                skip_rows_1 = 8
                
            # import data from the file, first 10 rows are strings and they are skipped
            data_from_file = np.loadtxt(file, delimiter = ',', skiprows = skip_rows_1, unpack = True)


            # just an x axis for plotting spectrum, array of wavelengths
            wavelengths_array = data_from_file[0]

            powermeter_readings.append(data_from_file[1][:166])

            # update list of indices by number of degrees at which current spectrum was taken
            indices_for_files_list.append(index_for_file)

    # changing list to numpy array        
    indices_for_files_array = np.array(indices_for_files_list)



    # saving data in table format
    all_data = pd.DataFrame(data=powermeter_readings, index=indices_for_files_array, columns=wavelengths_array)

    all_data = all_data.sort_index()
    
    return all_data


def find_closest_value(arr, value):
    diff = np.abs(arr - value)
    min_diff = np.min(diff)
    closest_indices = np.where(diff == min_diff)[0]
    return arr[closest_indices[0]]


def plotting_polarization(wavelengths_list, dataframe, no_polarizer_dataframe = None):
    
    wv_list = np.zeros( len(wavelengths_list) )
    
    k = 0
    for wavelength in wavelengths_list:
        wavelength_exact = find_closest_value( dataframe.columns.values , wavelength)
        wv_list[k] = wavelength_exact
        k = k + 1


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))


    for wv_exact in wv_list:
        ax1.plot( dataframe[wv_exact], '.-', label = f'{wv_exact}')

    ax1.set_title('Pixels at different wavelengths vs polarizartion angle')
    ax1.grid()

    if no_polarizer_dataframe is not None:
        for wv_exact in wv_list:
            ax2.plot( dataframe[wv_exact] / no_polarizer_dataframe.loc[1][wv_exact], '.-', label = f'{wv_exact}')
            ax2.set_title('Normalized to no polarizer')


    else:
        for wv_exact in wv_list:
            ax2.plot( dataframe[wv_exact] / np.mean(dataframe[wv_exact]), '.-', label = f'{wv_exact}')
            ax2.set_title('Normalized to mean')


    ax2.grid()
    ax2.legend()
    

    plt.tight_layout()


    plt.show()

####################################################################################################
def eliminate_noisy_data(data_array, noise_array, SN_ratio_value = None):
    """ 
    
    """

    sm, em = -1,-1
    cs = -1
    cl = 0


    SN_r = 20

    if SN_ratio_value is not None:
        SN_r = SN_ratio_value


    for ix, val in enumerate(noise_array*SN_r < np.abs(data_array)):
        if val == 1 and cs != -1:
            cl += 1
        elif val == 1 and cs == -1:
            cs = ix
            cl += 1
        if cl > em - sm:
            sm = cs
            em = sm + cl
        if val == 0:
            cs = -1
            cl = 0
    return np.arange(sm, em)



example_noise = np.ones(2048)*20

def eliminate_noisy_columns(dataframe, noise_array, SN_ratio = None):

    SN_R = 20

    if SN_ratio is not None:
        SN_R = SN_ratio
    
    first_array = eliminate_noisy_data(dataframe.loc[0].values, noise_array, SN_ratio_value = SN_R)

    for index in dataframe.index.values[1:]:
        
        new_array = eliminate_noisy_data(dataframe.loc[index].values, noise_array, SN_ratio_value = SN_R )
        first_array = list(set(first_array).intersection(new_array))

    return first_array


def denoisify_dataframe(dataframe, noise_array, SN_ratio = None):
    SN_r = 20

    if SN_ratio is not None:
        SN_r = SN_ratio

    first_array  = eliminate_noisy_columns(dataframe, noise_array, SN_ratio = SN_r)
    columns_to_keep = dataframe.columns.values[first_array]

    return dataframe[columns_to_keep]