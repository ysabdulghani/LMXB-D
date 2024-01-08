import torch   # Importing PyTorch for tensor operations
import matplotlib.pyplot as plt  # Importing matplotlib for plotting
import numpy as np  # Importing NumPy for numerical operations
import os  # Importing os for operating system interactions
import time  # Importing time for time-related functions
import h5py  # Importing h5py for interacting with H5 files
import boost_histogram as bh  # Importing os for operating system interactions
from scipy.stats import norm  # Importing scipy.stats for statistical functions
from scipy.special import erf  # Importing scipy.special for special mathematical functions
import urllib.request  # Importing urllib for handling URL requests
import scipy.ndimage  # Importing scipy.ndimage for image processing
import argparse  # Importing argparse for command-line argument parsing
from astropy.io import fits  # Importing astropy.io for astronomy-related IO operations
from multiprocessing import Pool,cpu_count  # Importing multiprocessing for parallel processing
from tqdm import tqdm # Importing tqdm for progress bars'
from matplotlib.colors import LinearSegmentedColormap # Importing matplotlib.colors for color map operations'

parser = argparse.ArgumentParser(description='Calculate mass/distance probabilities based on the probability distribution (MCMC distribution) of soft state model "ezdiskbb" normalization and the soft-to-hard transition period flux')
parser.add_argument('chainFilenames', type=str, help="Input .txt file containing the chain filenames in separate lines (include subdirectory if not in same path)")
parser.add_argument('--softonly', dest='soft', action='store_const', const=True, default=False, help="Flag to calculate for soft state only")
# Parse the argument
args = parser.parse_args()

torch.set_num_threads(2) # Best for multiprocessing

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

min_val, max_val = 0.3,1.0
n = 10
orig_cmap = plt.cm.Reds
colors = orig_cmap(np.linspace(min_val, max_val, n))
cmap_red = LinearSegmentedColormap.from_list("mycmap", colors)

min_val, max_val = 0.3,1.0
n = 10
orig_cmap = plt.cm.Blues
colors = orig_cmap(np.linspace(min_val, max_val, n))
cmap_blue = LinearSegmentedColormap.from_list("mycmap", colors)

def findParamIdx(chainColumnsNames, paramName):
    """
    Given a list of column names and a parameter name, this function returns the index of the column that contains the
    specified parameter name. The function converts all column names to lowercase to ensure consistent matching.

    Args:
    chainColumnsNames (list): A list of column names to search through.
    paramName (str): The name of the parameter to search for within the column names.

    Returns:
    int: The index of the column that contains the specified parameter name, or None if the parameter is not found.
    """
    paramName = paramName.lower()
    for i, colName in enumerate(map(str.lower, chainColumnsNames)):
        if paramName in colName:
            return i
    return None

def loadchainData(binSize=1,plotBinnedDistributions=False,softonly=False):  
    """
    Loads chain data from files and performs binning for memory efficiency and improved performance.

    Args:
    - binSize (int): The size of each bin for binning the norm and flux values. Default is 1 (no binning).
    - plotBinnedDistributions (bool): Flag indicating whether to plot the binned distributions. Default is False.

    Returns:
    - norm_binned (torch.Tensor): A tensor containing binned norm values.
    - fluxCGS_binned (torch.Tensor): A tensor containing binned flux values.

    Raises:
    - ValueError: If the "norm" parameter is not found in the soft chain file or the "flux" parameter is
                  not found in the transition chain file.
    """
    #Reading txt file containing the chain filenames
    with open(args.chainFilenames) as chains_f:
        chainsFiles = chains_f.read().splitlines()

    print("Soft state chain file is: ",chainsFiles[0])
    if not softonly:
        print("Transition point chain file is: ",chainsFiles[1])

    # Loading the chain file and reading norm
    hdul = fits.open(chainsFiles[0],memmap=True)
    data = hdul[1].data
    try:
        norm = data.field(findParamIdx(hdul[1].columns.names,'norm__5'))
    except:
        raise ValueError('Could not find "norm" in soft chain file')
        

    #Binning the norm for more efficient memory usage and better performance 
    norm.sort()
    norm_binned = torch.from_numpy(norm[:(norm.size // binSize) * binSize].reshape(-1, binSize).mean(axis=1))
    norm = None

    if plotBinnedDistributions:
        # Plot Norm distribution
        fig, ax1 = plt.subplots()
        count_d, bins_d, *temp = ax1.hist(norm_binned,bins='auto',density=True,ec="black",color='white')
        ax1.set_xlabel("Norm (diskbb)")
        ax1.set_ylabel("Probability")
        plt.show()
    
    if not softonly:
        # Loading the chain file and reading flux
        hdul = fits.open(chainsFiles[1],memmap=True)
        data = hdul[1].data
        try:
            log10flux = data.field(findParamIdx(hdul[1].columns.names,'flux'))
        except:
            raise ValueError('Could not find "flux" in transition chain file')
        fluxCGS = 10 ** log10flux

        # Binning the flux for more efficient memory usage and better performance 
        fluxCGS.sort()
        fluxCGS_binned = torch.from_numpy(fluxCGS[:(fluxCGS.size // binSize) * binSize].reshape(-1, binSize).mean(axis=1))
        log10flux = None
        fluxCGS = None

        if plotBinnedDistributions:
            # Plot Flux distribution
            fig, ax1 = plt.subplots()
            count_d, bins_d, *temp = ax1.hist(fluxCGS_binned,bins='auto',density=True,ec="black",color='white')
            ax1.set_xlabel("Flux (ergs/s)")
            ax1.set_ylabel("Probability")
            plt.show()
    
    if softonly:
        fluxCGS_binned = torch.tensor([0])
    
    return norm_binned,fluxCGS_binned

def idx_of_value_from_grid(grid,value,atol=1e-08,verbose=False):
    """
    Finds the index of a specified value in a grid array with a specified absolute tolerance.
    If no exact match is found, the tolerance is increased iteratively to find a close match.

    """
    index, = np.where(np.isclose(grid,value,rtol=1e-05, atol=atol))
    atol_tmp = atol
    while len(index) == 0:
        atol_tmp *= 10
        index, = np.where(np.isclose(grid,value,rtol=1e-05, atol=atol_tmp))
    if len(index) != 1:
        index = index + (index[-1]-index[0]) // 2
        if verbose==True:
            print("Warning: found more than grid value (for GR correction) that are equally close to the user specified value of %s. Taking the median value %s as the one to be closest." % (value,grid[index[0]]))
    return index

def search_for_inc_values_idx(i_grid,i_sample,verbose):
    i_indicies = []
    for i in i_sample:
        i_index = idx_of_value_from_grid(i_grid,i,verbose=verbose)
        i_indicies.append(i_index[0])
    return i_indicies

def quantile_index(pdf_values, q):
    """Find quantile index"""

    # Normalize the PDF values to make sure they sum to 1
    pdf_values = pdf_values / np.sum(pdf_values)
    
    # Calculate the cumulative sum of the PDF values
    cumulative_sum = np.cumsum(pdf_values)

    # Find the index where the cumulative sum exceeds the quantile
    quantile_index = np.searchsorted(cumulative_sum, q)
    
    return quantile_index

def calculate_contour_levels(H,levels):
    """Find contour levels. Adapted form corner.py/core.py file. https://corner.readthedocs.io/en/latest/"""

    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]
    V = np.empty(len(levels))
    for i, v0 in enumerate(levels):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except IndexError:
            V[i] = Hflat[0]
    V.sort()
    m = np.diff(V) == 0
    if np.any(m):
        print("Warning: Too few points to create valid contours")
    while np.any(m):
        V[np.where(m)[0][0]] *= 1.0 - 1e-4
        m = np.diff(V) == 0
    V.sort()
    return V

def skewed_gaussian(x, A, mu, sigma, alpha):
    # Skewed Gaussian equation
        return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) * (1 + erf(alpha * (x - mu) / (sigma * np.sqrt(2))))

def mass_weight_gauss(mass):
    # if device != torch.device("cpu"):
    #     mass = mass.cpu()
    # if mass < 3 or mass > 30: ## Might be re-implemented later
    #     return 0
    mass = mass.numpy()
    mean = 7.8  # OZEL ET AL 2010
    sigma = 1.2 # OZEL ET AL 2010
    return norm.pdf(mass,loc=mean,scale=sigma)

def get_total_correction_GR_and_Rin_Rg_ratio(inc,a,verbose=True,inc_sample_method='uniform',i_indicies=None):
    """
    Calculates the total correction factor and the Rin/Rg ratio based on the provided inclination angle and spin parameter.
    The correction factors are derived using General Relativity (GR) correction values from Greg Salvesen's repository.
    The function retrieves and uses data from the file 'gGR_gNT_J1655.h5' for these calculations.

    Args:
        inc (array-like): Inclination angles in degrees. This should contain at least the minimum and maximum of the desired inclination range.
        a (float, optional): Spin parameter, a dimensionless quantity representing the angular momentum of the black hole. Defaults to 0.
        verbose (bool, optional): If True, enables printing of warnings and additional information. Defaults to True.
        inc_sample_method (str, optional): Method for sampling inclination values, can be 'uniform' or 'isotropic'. Defaults to 'uniform'.
        i_indicies (list of int, optional): Pre-calculated inclination indices. If None, indices are calculated based on the inclination sample method.

    Returns:
        tuple: A tuple containing the following elements:
            - Torch tensor of the GR correction factors multiplied by the cosine of inclination and limb darkening.
            - Rin/Rg ratio value.
            - Selected spin value from the spin grid.
            - Array of selected inclination angles.
            - Indices of the selected inclination angles in the inclination grid.

    References:
        Greg Salvesen's repository for GR correction values: https://github.com/gregsalvesen/bhspinf
    """
    if not os.path.isfile('gGR_gNT_J1655.h5'):  
        urllib.request.urlretrieve("https://raw.githubusercontent.com/gregsalvesen/bhspinf/main/data/GR/gGR_gNT_J1655.h5","gGR_gNT_J1655.h5") ## Thanks Greg!!
    ## Copied from gregsalvesen/bhspinf/
    fh5      = 'gGR_gNT_J1655.h5'
    f        = h5py.File(fh5, 'r')
    a_grid   = f.get('a_grid')[:]  # [-]
    r_grid   = f.get('r_grid')[:]  # [Rg]
    i_grid   = f.get('i_grid')[:]  # [deg]
    gGR_grid = f.get('gGR')[:,:]   # gGR(r[Rg], i[deg])
    gNT_grid = f.get('gNT')[:]     # gNT(r[Rg])
    f.close()
    ##

    atol = 1e-08
    a_index, = np.where(np.isclose(a_grid,a,rtol=1e-05, atol=atol))
    atol_tmp = atol
    while len(a_index) == 0:
        atol_tmp *= 10
        a_index, = np.where(np.isclose(a_grid,a,rtol=1e-05, atol=atol_tmp))
    if len(a_index) != 1:
        a_index = a_index + (a_index[-1]-a_index[0]) // 2
        if verbose==True:
            print("Warning: found more than one spin grid value (for GR correction) that are equally close to the user specified spin value. Taking the median value %s as the one to be closest." % (a_grid[a_index[0]]))
    Rin_ratio = r_grid[a_index]
    
    inc_low_index, = np.where(np.isclose(i_grid,inc.min(),rtol=1e-05, atol=atol))
    inc_high_index, = np.where(np.isclose(i_grid,inc.max(),rtol=1e-05, atol=atol))
    
    if inc_sample_method == 'uniform':    
        if (inc_high_index[0]-inc_low_index[0]) <= 100:

            GR_correction = gGR_grid[a_index[0],inc_low_index[0]:inc_high_index[0]+1] * gNT_grid[a_index[0]]
            limb_darkening = (1/2 + (3/4)*np.cos(i_grid[inc_low_index[0]:inc_high_index[0]+1]*(np.pi/180)))
            cos_i = np.cos(i_grid[inc_low_index[0]:inc_high_index[0]+1]*(np.pi/180))
            selected_i = i_grid[inc_low_index[0]:inc_high_index[0]+1]
        else:
            GR_correction = gGR_grid[a_index[0],inc_low_index[0]:inc_high_index[0]+1:int((inc_high_index[0]-inc_low_index[0])/100)] * gNT_grid[a_index[0]]
            limb_darkening = (1/2 + (3/4)*np.cos(i_grid[inc_low_index[0]:inc_high_index[0]+1:int((inc_high_index[0]-inc_low_index[0])/100)]*(np.pi/180)))
            cos_i = np.cos(i_grid[inc_low_index[0]:inc_high_index[0]+1:int((inc_high_index[0]-inc_low_index[0])/100)]*(np.pi/180))
            selected_i = i_grid[inc_low_index[0]:inc_high_index[0]+1:int((inc_high_index[0]-inc_low_index[0])/100)]
    elif inc_sample_method == 'isotropic':
        if i_indicies==None:
            i_sample = np.arccos(np.linspace(np.cos(i_grid[inc_low_index]*(np.pi/180)),np.cos(i_grid[inc_high_index]*(np.pi/180)),101).ravel()) * (180/np.pi)
            i_indicies = search_for_inc_values_idx(i_grid,i_sample,verbose=False)
        selected_i = i_grid[i_indicies]
        GR_correction = gGR_grid[a_index[0],i_indicies] * gNT_grid[a_index[0]]
        limb_darkening = (1/2 + (3/4)*np.cos(i_grid[i_indicies]*(np.pi/180)))
        cos_i = np.cos(i_grid[i_indicies]*(np.pi/180))

    # GR_correction = 1 ##Uncomment for removing GR corrections
    # limb_darkening = 1 ##Uncomment for removing lim darkening

    return (torch.from_numpy(GR_correction*cos_i*limb_darkening),Rin_ratio[0],a_grid[a_index[0]],selected_i,i_indicies)

def calculate_distance(norm_binned, fluxCGS_binned, inc, mass, a, constraint='both',verbose=True,i_indicies=None):
    """
    Calculates the distance based on given constraints and parameters.

    Args:
        norm_binned (torch.Tensor): A tensor containing binned norm values.
        fluxCGS_binned (torch.Tensor): A tensor containing binned flux values.
        inc (float): Inclination value.
        mass (float): Mass value.
        a (float): Spin value.
        constraint (str, optional): Specifies the constraint type. Defaults to 'both'.
        verbose: set True for troubleshooting
        i_indicies = take already determined set of inclination values indicies from the GR corrections grid. 

    Returns:
        Tuple:
        - d_all (torch.Tensor): A tensor containing the calculated distance values.
        - corr_len (int): Length of the correction values.

    Raises:
        ValueError: If an invalid `constraint` type is provided.

    Example:
        >>> calculate_distance(norm_binned, fluxCGS_binned, 30.0, 3.5, 0.5, constraint='soft')
        (tensor([0.1, 0.2, 0.3, ...]), 5)

    """
    # Initializing constants needed for calculations
    G = 6.6743e-11  # SI units
    c = 2.998e8  # SI units
    kappa = 1.7
    m_sun = 1.989e30  # SI units
    edd_for_solar_mass = 1.3e38  # cgs
    cm_to_kpc = 3.24078e-22

    tic_calc = time.perf_counter()  # Timing the calculation

    # GR correction values
    corr, Rin_ratio, selected_a, selected_i, i_indicies = get_total_correction_GR_and_Rin_Rg_ratio(inc, a,verbose=False,inc_sample_method='isotropic',i_indicies=i_indicies)

    # # GPU
    # if torch.cuda.is_available():
    #     norm_binned = norm_binned.to(device)
    #     fluxCGS_binned = fluxCGS_binned.to(device) ## Will be re-added in future version
    #     corr = corr.to(device)

    # Distance distribution from constraint 1 (norm at soft state)
    if constraint == 'both' or constraint == 'soft':
        # Calculating distance distribution from constraint 1
        intermediate_soft_nocorr = ((Rin_ratio * G * m_sun * 1e-2) / ((kappa ** 2) * (c ** 2))) * torch.sqrt(1 / norm_binned)
        intermediate_soft = (intermediate_soft_nocorr.unsqueeze(1) * torch.sqrt(corr)).flatten()
        d_soft_constr = mass * intermediate_soft
        intermediate_soft_nocorr = None
        intermediate_soft = None

    # Distance distribution from constraint 2 (flux at transition point)
    if constraint == 'both' or constraint == 'trans':
        # Generating a distribution for eddFraction values
        N_eddFrac = len(corr)
        eddFrac_vector = torch.linspace(0.01, 0.04, N_eddFrac,device=device)
        
        # GPU
        # if torch.cuda.is_available():
        #     eddFrac_vector = eddFrac_vector.to(device) ## Will be re-added in future version

        # Calculating distance distribution from constraint 2
        intermediate_trans_noeddFrac = ((edd_for_solar_mass) / (4 * np.pi * fluxCGS_binned)) * (cm_to_kpc ** 2)
        intermediate_trans = (intermediate_trans_noeddFrac.unsqueeze(1) * eddFrac_vector).flatten()
        d_trans_constr = torch.sqrt(mass * intermediate_trans)
        intermediate_trans_noeddFrac = None
        intermediate_trans = None

    if constraint == 'both':
        # Concatenate distance from constraint 1 and 2
        d_all = torch.cat((d_soft_constr, d_trans_constr), dim=0)
        toc_calc = time.perf_counter()  # Timing calculation
        if verbose == True:
            print(f"Took {toc_calc - tic_calc:0.4f} seconds to do the calculations")
        return d_all,d_soft_constr,d_trans_constr, len(corr), i_indicies,selected_a
    elif constraint == 'soft':
        d_all = d_soft_constr
        d_soft_constr = None
        toc_calc = time.perf_counter()  # Timing calculation
        if verbose == True:
            print(f"Took {toc_calc - tic_calc:0.4f} seconds to do the calculations")
        return d_all, len(corr), i_indicies,selected_a
    elif constraint == 'trans':
        d_all = d_trans_constr
        d_soft_constr = None
        toc_calc = time.perf_counter()  # Timing calculation
        if verbose == True:
            print(f"Took {toc_calc - tic_calc:0.4f} seconds to do the calculations")
        return d_all, len(corr), i_indicies,selected_a
    else:
        raise ValueError("Invalid constraint type. Can only be 'both', 'soft', or 'trans'")

def process_mass(arguments):
    """
    Processes the mass distribution for a given set of parameters and computes histograms for distance calculations.

    Args:
        arguments (tuple): A tuple containing the following elements:
            - mass: Mass of the black hole.
            - a: Array of spin parameters.
            - args: Command line arguments containing flags and options.
            - norm_binned: Normalized binned data.
            - fluxCGS_binned: Binned flux data in CGS units.
            - inc: Inclination angles.
            - distance_bins: Bins for calculating distance histograms.
            - device: Computational device (CPU or GPU).

    Returns:
        tuple: A tuple containing the following elements:
            - weighted_hist: Weighted histogram considering the mass and optional distance constraints.
            - total_hist_soft: Total histogram for soft data.
            - total_hist_trans: Total histogram for transition data.
    """
    # Unpacking the arguments
    mass, a, args, norm_binned, fluxCGS_binned, inc, distance_bins, device = arguments

     # Iterating over the spin parameter
    for index in range(0,len(a)):
        # Handling the first index separatel
        if index == 0:
            if args.soft:
                # Calculating distance if soft only
                d_soft,_,i_indicies,_ = calculate_distance(norm_binned, fluxCGS_binned, inc, mass, a[index],verbose=False,constraint='soft')
                d_trans = torch.tensor([1])
            else:
                # Calculating distance
                _, d_soft, d_trans,_,i_indicies,_ = calculate_distance(norm_binned, fluxCGS_binned, inc, mass, a[index],verbose=False)
        # Calculating distance for the current spin parameter
        if args.soft:
            d_soft,_,_,_= calculate_distance(norm_binned, fluxCGS_binned, inc, mass, a[index],verbose=False,constraint='soft')
            d_trans = torch.tensor([1])
        else:
            _, d_soft, d_trans,_,_,_ = calculate_distance(norm_binned, fluxCGS_binned, inc, mass, a[index],verbose=False,i_indicies=i_indicies)
        # if device != torch.device("cpu"):
        #     hist_soft,_ = cp.histogram(cp.asarray(d_soft),bins=cp.asarray(distance_bins),density=False)
        #     hist_soft = cp.asnumpy(hist_soft)
        #     hist_trans,_ = cp.histogram(cp.asarray(d_trans), bins=cp.asarray(distance_bins),density=False)
        #     hist_trans = cp.asnumpy(hist_trans)
        # Histogram calculation
        if args.soft:
            hist_soft, _ = bh.numpy.histogram(d_soft.numpy(), bins=distance_bins.numpy(),threads=cpu_count(),density=False)
        else:
            hist_soft, _ = bh.numpy.histogram(d_soft.numpy(), bins=distance_bins.numpy(),threads=cpu_count(),density=False)
            hist_trans, _ = bh.numpy.histogram(d_trans.numpy(), bins=distance_bins.numpy(),threads=cpu_count(),density=False)

        if device != torch.device("cpu"):
                d_soft = d_soft.cpu()
                d_trans = d_trans.cpu()
        
        # Calculating weighted histograms
        weighted_hist_soft = hist_soft * mass_weight_gauss(mass)
        if not args.soft:
            weighted_hist_trans = hist_trans * mass_weight_gauss(mass)
        
        # Aggregating histograms
        if index == 0:
            total_hist_soft = weighted_hist_soft
            if not args.soft:
                total_hist_trans = weighted_hist_trans
            if not args.soft:
                weighted_hist = (hist_soft * hist_trans * mass_weight_gauss(mass))
            else:
                weighted_hist = (hist_soft * mass_weight_gauss(mass))

        else:
            total_hist_soft = total_hist_soft + weighted_hist_soft
            if not args.soft:
                total_hist_trans = total_hist_trans + weighted_hist_trans 
            if not args.soft:
                weighted_hist = weighted_hist + (hist_soft * hist_trans * mass_weight_gauss(mass))
            else:
                weighted_hist = weighted_hist + (hist_soft * mass_weight_gauss(mass))
        if args.soft:
            total_hist_trans = [1]
    # Return all necessary data
    return (weighted_hist, total_hist_soft, total_hist_trans)

def main():
    tic = time.perf_counter() #Timing the script
    num_processes = int(cpu_count()/2)
    print(torch.get_num_threads())
    # print(device)
    # Define/get parameters
    if args.soft:
        norm_binned, fluxCGS_binned = loadchainData(100,softonly=True) 
    else:
        norm_binned, fluxCGS_binned = loadchainData(100)
    inc = np.array([0,84])
    mass_min = 3
    mass_max = 20
    num_masses = 1000
    a_max = 0.998
    a_step = 0.01
    a = np.linspace(0,a_max,int(a_max/a_step)+1)
    distance_min = 0.01
    distance_max = 40.001
    distance_step = 0.01
    distance_bins = torch.arange(distance_min,distance_max,distance_step,device=device)

    # Generate a uniform distribution of masses
    masses = torch.linspace(mass_min, mass_max, num_masses,device=device)

    # Calculate histograms for each mass and apply weights
    weighted_histograms = []
    weighted_histograms_soft = []
    weighted_histograms_trans = []    

    # Mutliprocessing for faster execution
    with Pool(num_processes) as pool:
        args_for_pool = [(mass, a, args, norm_binned, fluxCGS_binned, inc, distance_bins, device) for mass in masses]
        all_results = list(tqdm(pool.imap(process_mass, args_for_pool), total=len(masses)))
        
    for result in all_results:
        weighted_histograms_per_mass, weighted_histograms_soft_per_mass, weighted_histograms_trans_per_mass = result

        # Combine the results
        weighted_histograms.append(weighted_histograms_per_mass)
        weighted_histograms_soft.append(weighted_histograms_soft_per_mass)
        weighted_histograms_trans.append(weighted_histograms_trans_per_mass)

    # if device != torch.device("cpu"):
    #     masses = masses.cpu()
    #     distance_bins = distance_bins.cpu()

    # Convert weighted histograms to numpy array
    if args.soft:
        weighted_histograms_np = np.stack(weighted_histograms_soft)
    else:
        weighted_histograms_np = np.stack(weighted_histograms)
      
    weighted_histograms_soft_np = np.stack(weighted_histograms_soft)
    weighted_histograms_trans_np = np.stack(weighted_histograms_trans)

    #Save arrays
    os.makedirs('prob_results',exist_ok=True)
    np.savez('prob_results/plot_data.npz', masses=masses.numpy(),distance_bins=distance_bins.numpy(),weighted_histograms_np=weighted_histograms_np, weighted_histograms_soft_np=weighted_histograms_soft_np, weighted_histograms_trans_np=weighted_histograms_trans_np)
    #Normalize
    weighted_histograms_np = weighted_histograms_np / (weighted_histograms_np.sum()*distance_step*((mass_max-mass_min)/(num_masses-1)))
    weighted_histograms_soft_np = weighted_histograms_soft_np / (weighted_histograms_soft_np.sum()*distance_step*((mass_max-mass_min)/(num_masses-1)))
    weighted_histograms_trans_np = weighted_histograms_trans_np / (weighted_histograms_trans_np.sum()*distance_step*((mass_max-mass_min)/(num_masses-1)))

    toc = time.perf_counter()
    print(f"Took {toc - tic:0.4f} seconds to finish probability calculations")

    #File to write summary of results stats
    resultsFilename = 'prob_results/probability_results'
    resultsFile = open(resultsFilename, 'w')

    print('Distance peak is at: %s' % ((distance_bins.numpy()[np.argmax(weighted_histograms_np.sum(axis=0)*((mass_max-mass_min)/(num_masses-1)))]+distance_bins.numpy()[np.argmax(weighted_histograms_np.sum(axis=0)*((mass_max-mass_min)/(num_masses-1)))+1])/2))
    resultsFile.write('Distance peak is at: %s\n' % ((distance_bins.numpy()[np.argmax(weighted_histograms_np.sum(axis=0)*((mass_max-mass_min)/(num_masses-1)))]+distance_bins.numpy()[np.argmax(weighted_histograms_np.sum(axis=0)*((mass_max-mass_min)/(num_masses-1)))+1])/2))
    print('The median distance is %.2f kpc. The 68%% confidence errors (upper,lower): (%.2f,%.2f)' %(distance_bins.numpy()[quantile_index(weighted_histograms_np.sum(axis=0)*((mass_max-mass_min)/(num_masses-1)),0.5)],distance_bins.numpy()[quantile_index(weighted_histograms_np.sum(axis=0)*((mass_max-mass_min)/(num_masses-1)),0.5)]-distance_bins.numpy()[quantile_index(weighted_histograms_np.sum(axis=0)*((mass_max-mass_min)/(num_masses-1)),0.16)],distance_bins.numpy()[quantile_index(weighted_histograms_np.sum(axis=0)*((mass_max-mass_min)/(num_masses-1)),0.84)]-distance_bins.numpy()[quantile_index(weighted_histograms_np.sum(axis=0)*((mass_max-mass_min)/(num_masses-1)),0.5)]))
    resultsFile.write('The median distance is %.2f kpc. The 68%% confidence errors (upper,lower): (%.2f,%.2f)\n' %(distance_bins.numpy()[quantile_index(weighted_histograms_np.sum(axis=0)*((mass_max-mass_min)/(num_masses-1)),0.5)],distance_bins.numpy()[quantile_index(weighted_histograms_np.sum(axis=0)*((mass_max-mass_min)/(num_masses-1)),0.5)]-distance_bins.numpy()[quantile_index(weighted_histograms_np.sum(axis=0)*((mass_max-mass_min)/(num_masses-1)),0.16)],distance_bins.numpy()[quantile_index(weighted_histograms_np.sum(axis=0)*((mass_max-mass_min)/(num_masses-1)),0.84)]-distance_bins.numpy()[quantile_index(weighted_histograms_np.sum(axis=0)*((mass_max-mass_min)/(num_masses-1)),0.5)]))
    if not args.soft:
        print('The median mass is %.2f solar masses. The 68%% confidence errors (upper,lower): (%.2f,%.2f)' %(masses.numpy()[quantile_index(weighted_histograms_np.sum(axis=1)*distance_step,0.5)],masses.numpy()[quantile_index(weighted_histograms_np.sum(axis=1)*distance_step,0.5)]-masses.numpy()[quantile_index(weighted_histograms_np.sum(axis=1)*distance_step,0.16)],masses.numpy()[quantile_index(weighted_histograms_np.sum(axis=1)*distance_step,0.84)]-masses.numpy()[quantile_index(weighted_histograms_np.sum(axis=1)*distance_step,0.5)]))
        resultsFile.write('The median mass is %.2f solar masses. The 68%% confidence errors (upper,lower): (%.2f,%.2f)\n' %(masses.numpy()[quantile_index(weighted_histograms_np.sum(axis=1)*distance_step,0.5)],masses.numpy()[quantile_index(weighted_histograms_np.sum(axis=1)*distance_step,0.5)]-masses.numpy()[quantile_index(weighted_histograms_np.sum(axis=1)*distance_step,0.16)],masses.numpy()[quantile_index(weighted_histograms_np.sum(axis=1)*distance_step,0.84)]-masses.numpy()[quantile_index(weighted_histograms_np.sum(axis=1)*distance_step,0.5)]))
    
    #Plot 1D histogram
    fig = plt.figure(num=1, clear=True)
    ax = fig.subplots()
    x = distance_bins[:-1] + ((distance_bins[1:] - distance_bins[:-1])/2)
    ax.plot(x,weighted_histograms_np.sum(axis=0)*((mass_max-mass_min)/(num_masses-1)),'k-',linewidth=1.0)
    if not args.soft: 
        ax.plot(x,weighted_histograms_soft_np.sum(axis=0)*((mass_max-mass_min)/(num_masses-1)),'r--',linewidth=1.0)
        ax.plot(x,weighted_histograms_trans_np.sum(axis=0)*((mass_max-mass_min)/(num_masses-1)),'b:',linewidth=1.0)
    ax.set_xlabel("Distance (kpc)")
    ax.set_ylabel("Probability Density (1/kpc)")
    plt.savefig('prob_results/dist_prob_plot.pdf', format='pdf', bbox_inches = 'tight')
    # plt.show()

    # Plot the 2D histogram
    fig = plt.figure(num=2, clear=True)
    ax1 = fig.subplots()

    levels = 1.0 - np.exp(-0.5 * np.arange(1,4,1) ** 2)

    if not args.soft:
        levels_soft_trans = 1.0 - np.exp(-0.5 * np.arange(1,7,1) ** 2)
        V_soft = calculate_contour_levels(weighted_histograms_soft_np,levels)
        V_trans = calculate_contour_levels(weighted_histograms_trans_np,levels)
        V_soft_filled = calculate_contour_levels(weighted_histograms_trans_np,levels_soft_trans)
        V_trans_filled = calculate_contour_levels(weighted_histograms_trans_np,levels_soft_trans)
        weighted_histograms_soft_np = scipy.ndimage.filters.gaussian_filter(weighted_histograms_soft_np, sigma=5)
        weighted_histograms_trans_np = scipy.ndimage.filters.gaussian_filter(weighted_histograms_trans_np, sigma=5)
        ax1.contourf(weighted_histograms_soft_np,V_soft_filled,cmap=cmap_red,extend="max",extent=[distance_bins.numpy().min(), distance_bins.numpy().max(), mass_min, mass_max],origin='lower')
        ax1.contourf(weighted_histograms_trans_np,V_trans_filled,cmap=cmap_blue,extend="max",extent=[distance_bins.numpy().min(), distance_bins.numpy().max(), mass_min, mass_max],origin='lower',alpha=0.7)
        ax1.contour(weighted_histograms_soft_np,V_soft,colors='black',linewidths=0.7,linestyles='dashed',extent=[distance_bins.numpy().min(), distance_bins.numpy().max(), mass_min, mass_max],origin='lower')
        ax1.contour(weighted_histograms_soft_np,V_trans,colors='black',linewidths=0.7,linestyles='dotted',extent=[distance_bins.numpy().min(), distance_bins.numpy().max(), mass_min, mass_max],origin='lower')

    V = calculate_contour_levels(weighted_histograms_np,levels)
    weighted_histograms_np = scipy.ndimage.filters.gaussian_filter(weighted_histograms_np, sigma=5)
    ax1.contourf(weighted_histograms_np,V,colors=['#C0C0C0', '#A0A0A0', '#808080'],extend="max",extent=[distance_bins.numpy().min(), distance_bins.numpy().max(), mass_min, mass_max],origin='lower')
    ax1.contour(weighted_histograms_np,V,colors='black',linewidths=0.7,linestyles='solid',extent=[distance_bins.numpy().min(), distance_bins.numpy().max(), mass_min, mass_max],origin='lower')

    ax1.set_xlabel('Distance (kpc)')
    ax1.set_ylabel(r'Mass ($M_{\odot}$)')
    plt.savefig('prob_results/mass_dist_prob_plot.pdf', format='pdf', bbox_inches = 'tight')
    # plt.show()

if __name__ == "__main__":
    main()