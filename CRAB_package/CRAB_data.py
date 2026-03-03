import pandas as pd
import numpy as np
import scipy.interpolate as interp
from scipy.optimize import curve_fit
from typing import Callable

from .functions import powerLaw, doublePowerLaw

class CRdata:
    def __init__(self, element: str, exp: str, df: pd.DataFrame, unit:str, flux_unit:str, widths:list):
        self.element = element
        self.exp_name = exp
        self.data = df.to_numpy(float)
        self.unit = unit
        self.flux_unit = flux_unit
        self.widths = widths
    
    @classmethod
    def from_db(cls, df: pd.DataFrame):
        data_objs = []

        # bin widths as list
        widths = list(df.iloc[:,3:5].values.astype(float))
        df.drop(df.columns[3:5], axis=1, inplace=True)
        # order as E/R, Flux, Error, Unit, Flux_unit, name, element
        df = df.iloc[:, [2,4,5,3,6,0,1]]

        # group by name and experiment
        for (name, element), group in df.groupby(['exp_name','element'],sort=False):
            data = group
            unit = data.iloc[0,4]
            f_unit = data['flux_unit'][0]
            data = data.iloc[:,0:3]
            data_objs.append(CRdata(element, name, data, unit, f_unit, widths))
            
        return data_objs


    @classmethod
    def from_fit(cls, og_obj, range:tuple, func:Callable, params:list, num_pts:int=100,):
        x = np.geomspace(range[0], range[1], num_pts)
        y = func(x, *params)
        data = np.column_stack((x,y))
        df = pd.DataFrame(data, columns=og_obj.df.columns[1:3])
        return CRdata(og_obj.element, func.__name__ + ' fit', df)
        
    # @property
    # def data(self):
    #     if self.widths is not None:
    #         return self.df.iloc[:,1:].to_numpy(float)
    #     else:
    #         return self.df.to_numpy(float)
    # @data.setter
    # def data(self, array):
    #     if self.widths is not None:
    #         self.df.iloc[:,1:] = array 
    #     else:
    #         self.df = array

    @staticmethod
    def read_data(dataname:str):
        data_init = pd.read_csv(dataname + '.csv', comment='@')
        return data_init
    @staticmethod
    def split_data(data_init:pd.DataFrame):
        final_data = []
        exp_col = data_init.columns[0]

        # Find where each new experiment starts
        for exp_name, group in data_init.groupby(exp_col, sort=False):
            # Reset index for clean slicing
            g = group.reset_index(drop=True)
    
            # New header for all groups 
            new_header = g.iloc[0, 2:]  # skip the experiment, element column
            data = g.iloc[1:, 2:]       # all rows after header, skipping columns 0 and 1
            data.columns = new_header

            # Apply header and clean up
            data = data.reset_index(drop=True)
            data = data.dropna(axis=1, how='all')  # drop columns where all elements are NaN

            # Save results
            final_data.append(CRdata(g.iloc[1, 1],exp_name,data))
        return final_data

    def selectData(self, columns:list[int], create_widths:bool=False):
        """selects data to be used for plotting

        Args:
            columns (list): columns to select for plotting
            mean (bool): whether to return the mean of the selected data

        """
        self.df = self.df.iloc[:,columns]
        if create_widths:
            self.widths = self.df.iloc[:,0:2].values.astype(float)
            self.df.drop(self.df.columns[0:2], axis=1, inplace=True)
            self.df.insert(loc=0, column='Bins', value=list(self.widths))

    def sliceData(self, range:tuple):
        data = self.data
        x = data[:,0]
        sel_idx = np.where((x>range[0]) & (x<range[1]))[0]
        sliced_data = data[sel_idx, :]
        new_df = pd.DataFrame(sliced_data, columns=self.df.columns[1:])
        new_df.insert(loc=0, column='Bins', value=list(self.widths[sel_idx]))
        newobj = CRdata(self.element, self.exp_name + f' on {range}', new_df)
        newobj.widths = self.widths[sel_idx]
        return newobj
        
    def errPrep(self, errors=1):
        selected_data = self.data
        sum = 0
        for i in range(errors):
            sum += (selected_data[:,-(i+1)])**2 
        err = np.sqrt(sum)
        selected_data = np.column_stack((selected_data[:,:-errors], err))
        self.df.drop(self.df.columns[-errors:], axis=1, inplace=True)
        self.df['Error'] = None
        self.data = selected_data

    def geoMean(self):
        widths = self.widths
        min = [w[0] for w in widths]
        max = [w[1] for w in widths]
        gmean = 10 ** ((np.log10(min) + np.log10(max)) / 2)
        self.df.insert(loc=1, column='Bin Avg', value=gmean)
    
    def prepData(self, errors:int=1, mean:bool=False, power:int=None):
        for i in range(errors):
            if self.df.iloc[1,-(i+1)][0] == '±':
                self.df.iloc[:,-(i+1)] = self.df.iloc[:, -(i+1)].str[1:]
        if mean:
            self.geoMean()
        self.errPrep(errors)
        if power:
            self.applyPower(power)
    
    def convertData(self, xconversion=1, yconversion=1):
        data = self.data
        data[:,0] = data[:,0] * xconversion
        data[:,1] = data[:,1] * yconversion
        data[:,2] = data[:,2] * yconversion
        self.data = data

    @staticmethod
    def thresholdSelect(x, y, err, threshold):
        # Create a boolean mask where x is greater than the threshold
        mask = x > threshold
        
        # Select x and y values based on the mask
        selected_x = x[mask]
        selected_y = y[mask]
        selected_err = err[mask]
        return selected_x, selected_y, selected_err
    
    def doubleBin(self, threshold=-1):  
        """
        Merge adjacent bins (pairwise) for bins whose x-value exceeds the threshold.
        Uses the width-weighted combination rule.
        Returns a NEW CRdata object.
        """

        if self.widths is None:
            raise ValueError("doubleBin requires self.widths (bin min/max arrays).")

        data = self.data
        x = data[:, 0]
        y = data[:, 1]
        err = data[:, 2]
        widths = self.widths
        emin = widths[:, 0]
        emax = widths[:, 1]
        dE = emax - emin

        # Identify their indices in the original arrays
        sel_idx = np.where(x > threshold)[0]

        merged_x = []
        merged_y = []
        merged_err = []
        merged_widths = []

        i = 0
        while i < len(sel_idx):
            idx = sel_idx[i]
            if i + 1 < len(sel_idx):
                idx2 = sel_idx[i + 1]
                # width-weighted merge
                new_emin = emin[idx]
                new_emax = emax[idx2]

                y_comb = (y[idx] * dE[idx] + y[idx2] * dE[idx2]) / (dE[idx] + dE[idx2])
                err_comb = np.sqrt((err[idx] * dE[idx])**2 +
                                (err[idx2] * dE[idx2])**2) / (dE[idx] + dE[idx2])
                x_comb = 10 ** ((np.log10(new_emin) + np.log10(new_emax)) / 2)

                merged_x.append(x_comb)
                merged_y.append(y_comb)
                merged_err.append(err_comb)
                merged_widths.append([new_emin, new_emax])
                i += 2
            else: # leftover bin
                merged_x.append(x[idx])
                merged_y.append(y[idx])
                merged_err.append(err[idx])
                merged_widths.append([emin[idx], emax[idx]])
                i += 1

        # ---- Now assemble final combined list, preserving bins below threshold ----
        keep_mask = x <= threshold

        final_x = np.concatenate([x[keep_mask], np.array(merged_x)])
        final_y = np.concatenate([y[keep_mask], np.array(merged_y)])
        final_err = np.concatenate([err[keep_mask], np.array(merged_err)])
        final_widths = np.vstack([widths[keep_mask], np.array(merged_widths)])

        # Create new CRdata object
        newdata = np.column_stack((final_x, final_y, final_err))
        newdf = pd.DataFrame(newdata, columns=self.df.columns[1:])
        newdf.insert(loc=0, column='Bins', value=list(final_widths))
        newobj = CRdata(self.element, self.exp_name + "_dbl", newdf)
        newobj.widths = final_widths

        return newobj

    def interpData(self, type, threshold=-1):

        def avgInterp(interpx, interpy, xpts):
            ypts = np.zeros(len(xpts))
            for i in range(len(xpts)):
                # find the closest points on either side
                lower_points = interpx[interpx < xpts[i]]
                upper_points = interpx[interpx > xpts[i]]
                if len(lower_points) == 0 or len(upper_points) == 0:
                    ypts[i] = np.nan  # or some other value indicating out of bounds
                    continue
                lower_idx = np.argmax(lower_points)
                upper_idx = len(lower_points)

                # average the y values at these points
                y_lower = interpy[lower_idx]
                y_upper = interpy[upper_idx]
                ypts[i] = 10 ** ((np.log10(y_lower) + np.log10(y_upper)) / 2)
            return ypts
        
        def CubicSplineInterp(interpx, interpy, xpts):
            cs = interp.CubicSpline(interpx, interpy)
            ypts = cs(xpts)
            return ypts
        
        def pchipInterp(interpx, interpy, xpts):
            pchip = interp.PchipInterpolator(interpx, interpy)
            ypts = pchip(xpts)
            return ypts

        selected_x, selected_y, selected_err = self.thresholdSelect(self.data[:,0], self.data[:,1], self.data[:,2], threshold)

        # Compute midpoints between each pair of original data points, estimate error there
        x_mid = 10 ** ((np.log10(selected_x[:-1]) + np.log10(selected_x[1:])) / 2)
        err_mid = 10 ** ((np.log10(selected_err[:-1]) + np.log10(selected_err[1:])) / 2)

        # Evaluate the cubic spline at those midpoints
        if type == 'cspline':
            y_mid = CubicSplineInterp(selected_x, selected_y, x_mid)
        elif type == 'average':
            y_mid = avgInterp(selected_x, selected_y, x_mid)
        elif type == 'pchip':
            y_mid = pchipInterp(selected_x, selected_y, x_mid)

        newdata = np.zeros((len(x_mid), 3))
        newdata[:,0] = x_mid
        newdata[:,1] = y_mid
        newdata[:,2] = err_mid
        newdf = pd.DataFrame(newdata, columns=self.df.columns)
        interpobj = CRdata(self.element, type + self.exp_name, newdf)
        return interpobj
    
    def doublePowerLawFit(self, p0:list=None):
        x = self.data[:,0]
        y = self.data[:,1]
        yerr = self.data[:,2]
        popt, pcov = curve_fit(doublePowerLaw, x, y, p0=p0, sigma=yerr)
        return popt, pcov
    
    def applyPower(self, power):
        data = self.data
        for i in range(data.shape[1]-1):
            data[:,i+1] *= data[:,0]**power
        self.data = data

    def __str__(self):
        return self.element + ', ' + self.exp_name + '\n' + self.data.__str__()
    
    def __repr__(self):
        return f'CRdata({self.element},{self.exp_name},{self.data})'