# -*- coding: utf-8 -*-
"""
Created on Thu Aug 8 2024

@author: Rohan Khamkar
"""

import time
from datetime import datetime
from tabulate import tabulate
import numpy as np
import helper as ks
import pyvisa
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PyPDF2 import PdfWriter, PdfReader
import serial
from reportlab.lib.pagesizes import letter, A4
from reportlab.pdfgen import canvas
from io import BytesIO
from datetime import datetime
import tkinter as tk
import tkinter.ttk as ttk
from scipy.interpolate import interp1d
import serial.tools.list_ports
from tkinter import Toplevel
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os

mux_commands = [
    "AA0100000000BB", "AA0100000100BB",  # Pixel 1
    "AA0101000000BB", "AA0101000100BB",  # Pixel 2
    "AA0102000000BB", "AA0102000100BB",  # Pixel 3
    "AA0103000000BB", "AA0103000100BB",  # Pixel 4
    "AA0104000000BB", "AA0104000100BB",  # Pixel 5
    "AA0105000000BB", "AA0105000100BB"   # Pixel 6
]

global keithley
global connected
pixel_data = {}

def serialopen(port):                    #Opens a serial port connection with specified parameters
    global ser
    try:
        ser=serial.Serial(port=port,baudrate=115200,bytesize=serial.EIGHTBITS,parity=serial.PARITY_NONE)
    except:
        pass

def connect_keithley():                  #Connects to a Keithley instrument using PyVISA library
    global keithley
    global connected
    rm = pyvisa.ResourceManager()
    print(rm.list_resources())
    
    connected=False
    try:
        keithley = rm.open_resource('ASRL4::INSTR')
        keithley.set_visa_attribute(pyvisa.constants.VI_ATTR_ASRL_BAUD, 9600)
        keithley.timeout = None
        print(keithley.query('*IDN?'))
        connected=True
    except Exception as e:
        connected=False

def send_command(keithley, command):
        '''

    Parameters 
    ----------
    command : The command that needs to be written to the keithley device to perform the specific function

    Returns
    -------
    None.

    '''
        response = ks.write(keithley, command)
        print(f"Sending command: {command}")      
        
def set_sensor_type(keithley, sensor_type):
    '''
    

    Parameters
    ----------
    sensor_type : Pass as CURR , VOLT sets the keithleys sensing mode to current sensor or voltage sensor
    Returns
    -------
    None.

    '''
    sensor_type = sensor_type.lower()
    if sensor_type == 'current' or sensor_type == 'curr' or sensor_type == 'c':
        send_command(keithley,':SENS:FUNC "CURR"')
        
    elif sensor_type == 'voltage' or sensor_type == 'volt' or sensor_type == 'v':
          send_command(keithley,':SENS:FUNC "VOLT"')
        
    else:
        print("Unknown sensor setting!")
        return None


def set_source_type(keithley, source_type):
    '''
    

    Parameters
    ----------
    sensor_type : Pass as CURR , VOLT sets the keithleys source mode to current or voltage sources
    Returns
    -------
    None.
    '''
    source_type = source_type.lower()
    if source_type == 'current' or source_type == 'curr' or source_type == 'c':
        send_command(keithley,':SOUR:FUNC CURRENT')
    elif source_type == 'voltage' or source_type == 'volt' or source_type == 'v':
        send_command(keithley,':SOUR:FUNC VOLT')
    else:
        print("Unknown source setting!")
        
# Function to send commands to Keithley device
def send_command(keithley, command):
    keithley.write(command)

def muxoperation(ser, value):
    f = bytes.fromhex(value)
    ser.write(f)
    time.sleep(2)

def select_pixel(ser, pixel_number, turn_off=False):
    '''
    Selects or turns off a specific pixel using the multiplexer.

    Parameters:
    - ser (serial.Serial): The serial port for multiplexer control.
    - pixel_number (int): The number of the pixel to be selected or turned off.
    - turn_off (bool): If True, the pixel will be turned off; otherwise, it will be selected.

    Returns:
    None
    '''
    
    if turn_off==False:
        print(f"Turning off Pixel {pixel_number}")
        # Here, you should send the "turn off" command specific to the pixel
        turn_off_command = f"AA010{pixel_number}000100BB"
        muxoperation(ser, turn_off_command)
    else:
        print(f"Selecting Pixel {pixel_number}")
        # Here, you should send the "turn on" command specific to the pixel
        turn_on_command = f"AA010{pixel_number}000000BB"
        print(turn_on_command)
        muxoperation(ser, turn_on_command)

class PlotNavigator:
    def __init__(self, master, vvalues_list, cvalues_list, pixel_numbers, area, radiation):
        self.master = master
        self.vvalues_list = vvalues_list
        self.cvalues_list = cvalues_list
        self.pixel_numbers = pixel_numbers
        self.area = area
        self.radiation = radiation
        self.current_index = 0
        
        self.fig = Figure(figsize=(6, 4))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        # Buttons for navigation
        self.prev_button = tk.Button(master, text="Previous", command=self.show_previous)
        self.prev_button.pack(side=tk.LEFT)
        
        self.next_button = tk.Button(master, text="Next", command=self.show_next)
        self.next_button.pack(side=tk.RIGHT)
        
        self.show_plot(self.current_index)
    
    def show_plot(self, index):
        self.ax.clear()
        # Reuse the plot_new function for plotting
        plot_new(self.vvalues_list[index], self.cvalues_list[index], self.pixel_numbers[index], self.ax, self.area, self.radiation)
        self.canvas.draw()
    
    def show_previous(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_plot(self.current_index)
    
    def show_next(self):
        if self.current_index < len(self.vvalues_list) - 1:
            self.current_index += 1
            self.show_plot(self.current_index)

def create_output_folder(username):
    timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    folder_name = f"{username}_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)
    return folder_name

def mux(keithley, ser, pixel_count, start, stop, step, delay, username, area, direction, radiation, output_folder):
    '''
    Coordinates measurements for multiple pixels by switching between them, performing sweeps, and saving data to text files.

    Parameters:
    - keithley (pyvisa.Resource): The Keithley device resource.
    - ser (serial.Serial): The serial port for multiplexer control.
    - pixel_count (int): The total number of pixels to be measured.
    - num_sweeps (int): The number of voltage sweeps to perform.
    - start (float): The starting voltage for the sweeps.
    - stop (float): The ending voltage for the sweeps.
    - step (float): The voltage step size.

    Saves data for each pixel in separate text files.
    '''
    pdf_file_path = os.path.join(output_folder, 'plots.pdf')
    all_voltage_values = []
    all_current_values = []
    pixel_numbers = []
    num_rows = 3
    num_cols = 2
    a4_width_inches = 8.27
    a4_height_inches = 11.69
    
    # Initialize figures for IV plots and current density vs V plots
    fig_iv, ax_iv = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(8, 6))
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    
    fig_density, ax_density = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(8, 6))
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    
    with PdfPages(pdf_file_path) as pdf:
        # Sweep and plot IV curves
        for pixel_number in range(pixel_count):
            # Turn on the current pixel
            select_pixel(ser, pixel_number, turn_off=True)
            
            row_index = pixel_number // num_cols
            col_index = pixel_number % num_cols
            
            # Continue with voltage sweeps and data collection for the current pixel
            voltage_values, current_values, time_values = sweep(keithley, start, stop, step, delay, direction)
            #voltage_values_d, current_values_d, time_values_d = sweep(keithley, start, stop, step, delay, 1)
            
            combined_voltage_values = voltage_values #+ voltage_values_d
            combined_current_values = current_values #+ current_values_d
            
            # Accumulate data
            all_voltage_values.append(combined_voltage_values)
            all_current_values.append(combined_current_values)
            pixel_numbers.append(pixel_number)

            # Save data for the current pixel in a text file
            save_data_to_file(pixel_number, combined_voltage_values, combined_current_values, username, output_folder)
            
            # Plot IV curve on the first page
            plot_new(combined_voltage_values, combined_current_values, pixel_number, ax_iv[row_index, col_index], area, radiation)
            
            # Plot current density vs V on the second page
            driftcurr_vs_V(combined_voltage_values, combined_current_values, pixel_number, ax_density[row_index, col_index], area)
            
            # Turn off the current pixel after completing its measurements
            select_pixel(ser, pixel_number, turn_off=False)

        # Adjust figure size and save pages to PDF
        fig_iv.set_size_inches(a4_width_inches, a4_height_inches)
        pdf.savefig(fig_iv)
        plt.close(fig_iv)
        
        fig_density.set_size_inches(a4_width_inches, a4_height_inches)
        pdf.savefig(fig_density)
        plt.close(fig_density)

    plot_navigator_window = tk.Toplevel()
    plot_navigator = PlotNavigator(plot_navigator_window, all_voltage_values, all_current_values, pixel_numbers, area, radiation)

    return pdf_file_path

def manual_sweeping(keithley, ser, pixel_number, sweep_count, start, stop, step, delay, username, area, direction, radiation, output_folder):
    '''
    Performs multiple voltage sweeps for a specific pixel, turning it on and off, and saving data to a text file and PDF.

    Parameters:
    - keithley (pyvisa.Resource): The Keithley device resource.
    - ser (serial.Serial): The serial port for multiplexer control.
    - pixel_number (int): The number of the pixel to be measured.
    - sweep_count (int): Number of times the pixel is swept.
    - start (float): The starting voltage for the sweeps.
    - stop (float): The ending voltage for the sweeps.
    - step (float): The voltage step size.
    - delay (float): Delay in milliseconds between steps.
    - username (str): The username for file naming.
    - area (float): The area of the pixel for density calculations.
    - direction (str): The direction of the sweep ("up", "down", or "both").

    Returns:
    - pdf_file_path (str): The path to the saved PDF file containing the plots.
    '''
    pdf_file_path = os.path.join(output_folder, f'{username}_pixel_{pixel_number}_sweep.pdf')
    
    # Lists to store all sweep data
    all_voltage_values = []
    all_current_values = []
    
    # Perform sweeps multiple times and collect data
    for i in range(sweep_count):
        # Turn on the specific pixel
        select_pixel(ser, pixel_number, turn_off=True)
        
        # Perform the voltage sweep and collect data
        voltage_values, current_values, time_values = sweep(keithley, start, stop, step, delay, direction)
        
        # Store the data from each sweep
        all_voltage_values.append(voltage_values)
        all_current_values.append(current_values)
        
        # Save data for the specific pixel in a text file (optional)
        save_data_to_file(pixel_number, voltage_values, current_values, username + f'_sweep_{i+1}', output_folder)
        
        # Turn off the pixel after completing its measurements
        select_pixel(ser, pixel_number, turn_off=False)
    
    # Plot all sweeps on a single IV curve
    fig_iv, ax_iv = plt.subplots(figsize=(8, 6))
    for i in range(sweep_count):
        plot_new(all_voltage_values[i], all_current_values[i], f'Pixel {pixel_number} Sweep {i+1}', ax_iv, area, radiation)
    
    # Set title and labels
    ax_iv.set_title(f'Pixel {pixel_number} IV Curve - {sweep_count} Sweeps Combined')
    ax_iv.set_xlabel("Voltage (mV)")
    ax_iv.set_ylabel("Current (mA)")
    
    # Save the combined plot in the PDF
    with PdfPages(pdf_file_path) as pdf:
        fig_iv.set_size_inches(8.27, 11.69)  # A4 size dimensions
        pdf.savefig(fig_iv)
        plt.close(fig_iv)
    
    return pdf_file_path

def save_data_to_file(pixel_number, voltage_values, current_values, username, output_folder):
    # Convert voltage from V to mV and current from A to mA
    voltage_values_mV = [v * 1000 for v in voltage_values]
    current_values_mA = [c * 1000 for c in current_values]

    # Current date in yyyy.mm.dd format
    current_date = datetime.now().strftime('%Y.%m.%d')

    # Define the file name based on the pixel number and username
    file_name = f'{current_date}_{username}_{pixel_number}_data.txt'
    file_path = os.path.join(output_folder, file_name)

    # Create a data table with the converted values
    data_table = list(zip(voltage_values_mV, current_values_mA))

    # Open the file in write mode and save the data
    with open(file_path, 'w') as file:
        file.write("Voltage (mV)\tCurrent (mA)\n")
        file.write(tabulate(data_table, tablefmt='plain', floatfmt=('.6f', '.6e')))
            
def plot_new(vvalues, cvalues, pixel_number, ax, area, radiation):
    global pixel_data
    # Convert voltage from V to mV and current from A to mA
    vvalues_mV = np.array(vvalues) * 1000
    cvalues_mA = np.array(cvalues) * -1000

    # Plot the data
    ax.plot(vvalues_mV, cvalues_mA, label='IV Curve')

    # Interpolation to find Voc (I = 0)
    interpolator_Voc = interp1d(cvalues, vvalues, fill_value="extrapolate")
    Voc = interpolator_Voc(0) * 1000  # Convert to mV

    # Interpolation to find Isc (V = 0)
    interpolator_Isc = interp1d(vvalues, cvalues, fill_value="extrapolate")
    Isc = interpolator_Isc(0) * -1000  # Convert to mA

    # Calculate power for each point
    P = vvalues_mV * cvalues_mA

    # Find the maximum power point
    Pmax_index = np.argmax(P)
    Vmax = vvalues_mV[Pmax_index]
    Imax = cvalues_mA[Pmax_index]
    Pmax = P[Pmax_index]

    # Plot nominal power point
    Pnominal = Voc * Isc  # Nominal power microW
    
    # Calculate Fill Factor (FF) and Efficiency
    FF = Pmax / Pnominal if Pnominal != 0 else 0
    efficiency = ((Pmax) / (radiation * area)) if radiation != 0 and area != 0 else 0  

    # Update the global dictionary
    pixel_data[pixel_number] = {
        'Pmax': Pmax,
        'Vmax': Vmax,
        'Imax': Imax, 
        'Pnom': Pnominal,
        'Voc': Voc,
        'Isc': Isc,
        'FF': FF,
        'Efficiency': efficiency
    }

    # Plot maximum power point
    ax.plot(Vmax, Imax, 'yo', label=f'Pmax: {Pmax:.4f} µW\n(Vm={Vmax:.4f} mV, Im={Imax:.4f} mA)', markersize=8)

    # Plot nominal power point
    ax.plot(Voc, Isc, 'ro', label=f'Pnom: {Pnominal:.4f} µW\n(Voc={Voc:.4f} mV, Isc={Isc:.4f} mA)', markersize=8)

    # Add FF and efficiency to the plot label
    ax.plot([], [], ' ', label=f'FF: {FF:.4f}\nEff: {efficiency:.2f}%')

    # Set title and labels
    ax.set_title(f'Pixel {pixel_number} IV Curve')
    ax.set_xlabel("Voltage (mV)")
    ax.set_ylabel("Current (mA)")
    ax.grid(True)

    # Add legend
    ax.legend()
    ax.grid(True)

def driftcurr_vs_V(vvalues, cvalues, pixel_number, ax, area):
    # Convert voltage to mV and current to mA
    vvalues_mV = np.array(vvalues) * 1000
    cvalues_mA = np.array(cvalues) * -1000
    
    # Calculate current density (mA/cm²)
    current_density = cvalues_mA / area
    
    # Calculate power for each point
    P = vvalues_mV * current_density
    
    # Find the maximum power point (Pmax)
    Pmax_index = np.argmax(P)
    Vmax = vvalues_mV[Pmax_index]
    Jmax = current_density[Pmax_index]  # Current density at maximum power point
    Pmax = P[Pmax_index]

    # Plot the IV data (voltage vs current density)
    ax.plot(vvalues_mV, current_density, label=f'Pixel {pixel_number} IV Curve')

    # Mark the Vmax and Jmax point (corresponding to Pmax)
    ax.plot(Vmax, Jmax, 'ro', label=f'Pmax: {Pmax:.4f} µW\n(Vmax: {Vmax:.4f} mV, Jmax: {Jmax:.4f} mA/cm²)', markersize=8)

    # Set title and labels
    ax.set_title(f'Pixel {pixel_number} Drift Current vs Voltage')
    ax.set_xlabel("Voltage (mV)")
    ax.set_ylabel("Current Density (mA/cm²)")
    ax.grid(True)
    
    # Add legend to indicate Vmax, Jmax, and Pmax
    ax.legend()

def sweep(keithley, start_mV, stop_mV, step_mV, delay_mS, direction):
    '''
    This function performs a voltage sweep on the Keithley device.

    Parameters:
    - keithley (pyvisa.Resource): The Keithley device resource.
    - start (float): The starting voltage for the sweeps.
    - stop (float): The ending voltage for the sweeps.
    - step_mV (float): The voltage step size in mV.
    - delay_mS (float): Delay in milliseconds between steps
    - num_sweeps (int): The number of voltage sweeps to perform.

    Returns:
    Lists of voltage and current values for each sweep.
    '''
    # Convert from mV to V
    start = start_mV/1000.0
    stop = stop_mV/1000.0
    step = step_mV/1000.0

    # Convert delay from milliseconds to seconds
    delay_s = delay_mS / 1000.0
    
    #sweep_points = int((stop - start) / step) + 1
    #num_trigs = sweep_points
    if direction == "Up":
        voltages = [start + step * i for i in range(int((stop - start) / step) + 1)]
    elif direction == "Down":
        voltages = [stop - step * i for i in range(int((stop - start) / step) + 1)]
    elif direction == "Both":
        sweep_up = [start + step * i for i in range(int((stop - start) / step) + 1)]
        sweep_down = [stop - step * i for i in range(int((stop - start) / step) + 1)]
        voltages = sweep_up + sweep_down[1:]
    else:
        raise ValueError("Invalid direction")

    #all_voltage_values = []
    #all_current_values = []
    #all_time_values=[]
    voltage_data = []
    current_data = []
    time_data=[]

   # for i in range(num_sweeps):
    send_command(keithley, '*RST')  # Reset the device to default settings
    send_command(keithley, ':FORM:ELEM VOLT,CURR,TIME')
    send_command(keithley, ':SOUR:FUNC VOLT')
    send_command(keithley, ':SENS:FUNC "CURR"')
    #send_command(keithley, ':SENS:CURR:UNIT MA')  # Set current measurement units to microamperes
    send_command(keithley, ':SENS:CURR:PROT 0.3')
    send_command(keithley, ':SOUR:VOLT:MODE FIX')
    send_command(keithley, ':OUTP ON')

    print('Sweep started')

    # Perform the sweep with delay between steps
    #if direction == 0:
        #voltages = [start + step * i for i in range(num_trigs)]
    #else:
        #voltages = [stop - step * i for i in range(num_trigs)]
        
    for voltage in voltages:
        send_command(keithley, f':SOUR:VOLT {voltage}')
        time.sleep(delay_s)
        response = keithley.query(':READ?')
        data = [float(val) for val in response.split(',')]
        voltage_data.append(data[0])
        current_data.append(data[1])
        time_data.append(data[2])
        
    print('Sweep done')
    
    send_command(keithley, ':OUTP OFF')

    return voltage_data, current_data, time_data

def single_pixel_sweep(keithley, ser, pixel_number, voltages, delay_mS):
    # Convert delay from milliseconds to seconds
    delay_s = delay_mS / 1000.0
    
    voltage_data = []
    current_data = []
    
    select_pixel(ser, pixel_number, turn_off=True)
    
    send_command(keithley, '*RST')  # Reset the device to default settings
    send_command(keithley, ':FORM:ELEM VOLT,CURR,TIME')
    send_command(keithley, ':SOUR:FUNC VOLT')
    send_command(keithley, ':SENS:FUNC "CURR"')
    #send_command(keithley, ':SENS:CURR:UNIT MA')  # Set current measurement units to microamperes
    send_command(keithley, ':SENS:CURR:PROT 0.3')
    send_command(keithley, ':SOUR:VOLT:MODE FIX')
    send_command(keithley, ':OUTP ON')

    print('Sweep started')
        
    for voltage in voltages:
        send_command(keithley, f':SOUR:VOLT {voltage}')
        time.sleep(delay_s)
        response = keithley.query(':READ?')
        data = [float(val) for val in response.split(',')]
        voltage_data.append(data[0])
        current_data.append(data[1])

        print(f"Voltage: {data[0]:.4f} V, Current: {data[1]:.6f} A")
        
    print('Sweep done')
    select_pixel(ser, pixel_number, turn_off=False)
    
    send_command(keithley, ':OUTP OFF')

    return voltage_data, current_data

def find_and_store_max_Pmax_pixel(area, radiation):
    global pixel_data
    
    if not pixel_data:
        print("No pixel data available.")
        return None, None, None, None, None, None, None, None
    
    best_pixel = max(pixel_data, key=lambda k: pixel_data[k]['Efficiency'])

    # Extract the relevant values for the best pixel
    Pmax = pixel_data[best_pixel]['Pmax']
    Pnom = pixel_data[best_pixel]['Pnom']
    Imax = pixel_data[best_pixel]['Imax']
    Vmax = pixel_data[best_pixel]['Vmax']
    Isc = pixel_data[best_pixel]['Isc']
    Voc = pixel_data[best_pixel]['Voc']
    FF = pixel_data[best_pixel]['FF']
    efficiency = pixel_data[best_pixel]['Efficiency']
    
    print(f"Best Pixel: {best_pixel} with Efficiency: {efficiency:.2f}%")
    print(f"Pmax: {Pmax:.4f} µW, Imax: {Imax:.4f} mA, Vmax: {Vmax:.4f} mV, FF: {FF:.4f}")

    return best_pixel, Pmax, Pnom, Voc, Isc, Imax, Vmax, FF, efficiency
    
def mppt(keithley, ser, pixel_number, Pmax, Pnom, Isc, Voc, Imax, Vmax, FF, efficiency, voltage_change, end_voltage_differ, runtime_minutes, delay_mS, username, radiation, area, output_folder):
    pixel_number = pixel_number
    Imax = Imax
    Vmax = Vmax
    Pmax = Pmax
    Pnom= Pnom
    FF= FF
    efficiency = efficiency

    pdf_file_path = os.path.join(output_folder, f'{username}_pixel_{pixel_number}_mppt.pdf')
    text_file_path = os.path.join(output_folder, f'{username}_pixel_{pixel_number}_mppt_log.txt')

    start_voltage = (Voc) / 1000.0 
    voltage_change = abs(voltage_change) / 1000.0
    # Lists to track Pmax and iteration numbers
    pmax_values = []
    ff_values = []
    efficiency_values = []
    elapsed_time_values = []
    start_time = time.time()
    runtime_seconds = runtime_minutes * 60
    iv_curves = []
    iteration = 0

    with open(text_file_path, 'w') as log_file:
        log_file.write("Elapsed Time (minutes), Pmax (µW), Vmax (mV), Imax (mA), FF, Efficiency (%)\n")
        
        while True:
            # Check if the elapsed time exceeds the runtime limit
            current_time = time.time()
            elapsed_time = current_time - start_time
            elapsed_time_minutes = elapsed_time / 60
            if elapsed_time > runtime_seconds:
                print(f"Time limit of {runtime_minutes} minutes reached. Exiting the loop.")
                break
        
            end_voltage = (Vmax - end_voltage_differ) / 1000.0

            if start_voltage>end_voltage:
                voltages = [start_voltage - voltage_change * i for i in range(int(abs(start_voltage - end_voltage) / voltage_change) + 1)]
                voltage_values, current_values = single_pixel_sweep(keithley, ser, pixel_number, voltages, delay_mS)
                
                # Calculate P1 and P2
                voltage_mV = np.array(voltage_values) * 1000
                current_mA = np.array(current_values) * -1000
                powers = [v * c for v, c in zip(voltage_mV, current_mA)]
                max_power_index = np.argmax(powers)
                new_Pmax = powers[max_power_index]
                new_Vmax = voltage_mV[max_power_index]
                new_Imax = current_mA[max_power_index]
        
                Pmax = new_Pmax 
                Vmax = new_Vmax
                Imax = new_Imax 

                FF = Pmax / Pnom if Pnom != 0 else 0
                efficiency = ((Pmax) / (radiation * area)) if radiation != 0 and area != 0 else 0
                voltage_change = voltage_change
                pmax_values.append(Pmax)
                ff_values.append(FF)
                efficiency_values.append(efficiency)
                elapsed_time_values.append(elapsed_time_minutes)

                iv_curves.append((voltage_mV, current_mA, Vmax, Imax, Pmax))

                print(f"Elapsed time {elapsed_time_minutes:.2f} minutes: Pmax: {new_Pmax} µW, Vmax: {new_Vmax} mV, Imax: {new_Imax} mA, FF: {FF}, Efficiency: {efficiency}%")

            else:
                print(f'Out of range, start voltage: {start_voltage} and end voltage: {end_voltage}')
        
    with PdfPages(pdf_file_path) as pdf:
        # Plot Pmax vs elapsed time
        fig, ax1 = plt.subplots(figsize=(8, 6))
        ax1.plot(elapsed_time_values, pmax_values, marker='o', linestyle='-', color='b')
        for i, (x, y) in enumerate(zip(elapsed_time_values, pmax_values)):
            ax1.annotate(f'{y:.2f} µW',  # Label for Pmax
                         (x, y),
                         textcoords="offset points",  # Positioning
                         xytext=(0, -15),  # Adjust to position below the point
                         ha='center', fontsize=8, color='blue')  # Customize appearance
        ax1.set_xlabel('Elapsed Time (minutes)')
        ax1.set_ylabel('Pmax (µW)')
        ax1.set_title('Pmax vs. Elapsed Time')
        ax1.grid(True)
        fig.set_size_inches(8.27, 11.69)  # A4 size dimensions
        pdf.savefig(fig)
        plt.close(fig)

        # Plot IV curves with Vmax and Pmax highlighted
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        for i, (voltage_mV, current_mA, Vmax, Imax, Pmax) in enumerate(iv_curves):
            ax2.plot(voltage_mV, current_mA, label=f'Iteration {i} (Pmax: {Pmax:.2f} µW)')
            ax2.plot(Vmax, Imax, 'ro')  # Mark the point corresponding to Vmax

        ax2.set_xlabel('Voltage (mV)')
        ax2.set_ylabel('Current (mA)')
        ax2.set_title('IV Curves with Vmax and Pmax Highlighted')
        ax2.grid(True)

        # Save the IV curve plot to the PDF
        fig2.set_size_inches(8.27, 11.69)  # A4 size dimensions
        pdf.savefig(fig2)
        plt.close(fig2)

    return pdf_file_path, text_file_path

def run_start_up_commands(keithley):
    '''
    
     Function to run when the device is startup. It sets all device paramters to its default value

    Returns
    -------
    None.

    '''
    for com in start_up_commands:
        send_command(keithley,com)
        time.sleep(.01)

def add_page_heading(pdf_path, title, author, other_details,current_date):
    packet = BytesIO()
    can = canvas.Canvas(packet, pagesize=A4)

    # Set font and size for the heading
    can.setFont("Helvetica", 16)

    # Add heading details
    can.drawString(160, 550, f"Title: {title}")
    can.drawString(160, 530, f"Author: {author}")
    can.drawString(160, 510, f"Other Details: {other_details}")
    can.drawString(160, 490, f"Date: {current_date}")
    # Save the canvas
    can.save()

    # Move the packet cursor to the beginning
    packet.seek(0)

    # Create a new PDF with the heading
    new_pdf = PdfReader(packet)
    existing_pdf = PdfReader(open(pdf_path, "rb"))
    output_pdf = PdfWriter()

    # Add the heading page
    heading_page = new_pdf.pages[0]
    output_pdf.add_page(heading_page)

    # Add the existing pages
    for i in range(len(existing_pdf.pages)):
        page = existing_pdf.pages[i]
        output_pdf.add_page(page)

    # Write the combined PDF to a new file
    with open("final_plots.pdf", "wb") as final_pdf:
        output_pdf.write(final_pdf)


#the various startup commands that need to be excecuted at the start of the system. 
start_up_commands = ["*RST",
                     ":SYST:TIME:RES:AUTO 1",
                     ":SYST:BEEP:STAT 1",
                     ":SOUR:FUNC CURR",
                     ":SENS:FUNC:CONC OFF",
                     ":SENS:AVER:STAT OFF",
                     ":SENS:CURR:NPLC 0.01",
                     ":SENS:VOLT:NPLC 0.01",
                     ":SENS:RES:NPLC 0.01",
                     ":SENS:FUNC 'VOLT'",
                     ":SENS:VOLT:RANG 1e1",
                     ":TRIG:DEL 0.0",
                     ":SYST:AZER:STAT OFF",
                     ":SOUR:DELAY 0.0",
                     ":DISP:ENAB ON"]

#!/usr/bin/python3


class SolarCellTestApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Solar Cell Test")
        self.geometry("400x800")
        
        # Initialize variables
        self.user_name = tk.StringVar()
        self.com_port = tk.StringVar()
        self.start_voltage = tk.DoubleVar(value=-200)
        self.stop_voltage = tk.DoubleVar(value=1200)
        self.sweep_voltage = tk.DoubleVar(value=100)
        self.sweep_delay = tk.DoubleVar(value=10)
        self.area_value = tk.DoubleVar(value=0.089)
        self.radiation_value = tk.DoubleVar(value=1000)
        self.pixel_number = tk.IntVar(value=1)
        self.sweep_count = tk.IntVar(value=1)
        #self.on_off = tk.StringVar(value="Off")
        self.direction = tk.StringVar(value="Up")
        #MPPT
        self.voltage_change = tk.DoubleVar(value=10)  
        self.end_voltage_differ = tk.DoubleVar(value=5)  
        self.time_period = tk.IntVar(value=30)  

        # Create canvas, scrollbar, and inner frame
        self.canvas = tk.Canvas(self, borderwidth=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Populate the inner frame with widgets
        self.create_widgets()

    def create_widgets(self):
        # Solar Cell Test section
        frame = ttk.LabelFrame(self.scrollable_frame, text="Solar Cell Test")
        frame.pack(padx=10, pady=10, fill="x", expand=True)

        # Username Label and Entry
        ttk.Label(frame, text="User Name:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.username_entry = ttk.Entry(frame, textvariable=self.user_name)
        self.username_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew", columnspan=2)

        # Connect Keithley Button
        self.connect_keithley_button = ttk.Button(frame, text="Connect Keithley", command=self.connect_keithley)
        self.connect_keithley_button.grid(row=1, column=1, padx=5, pady=5, sticky="w", columnspan=1)

        # COM Port Label and Combobox
        ttk.Label(frame, text="COM PORT:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        available_ports = self.get_com_ports()
        self.com_port_combobox = ttk.Combobox(frame, textvariable=self.com_port, values=available_ports)
        self.com_port_combobox.grid(row=2, column=1, padx=5, pady=5, sticky="ew", columnspan=2)
        self.com_port_combobox.bind("<<ComboboxSelected>>", self.comval)

        # Connect Mux Button
        self.connect_mux_button = ttk.Button(frame, text="Connect Mux", command=self.connect_multiplexer)
        self.connect_mux_button.grid(row=3, column=1, padx=5, pady=5, sticky="w", columnspan=1)

        # Disconnect Keithley Button
        self.disconnect_keithley_button = ttk.Button(frame, text="Disconnect Keithley", command=self.disconnect_keithley)
        self.disconnect_keithley_button.grid(row=4, column=1, padx=5, pady=5, sticky="w", columnspan=1)

        # All 6 Pixel section
        frame = ttk.LabelFrame(self.scrollable_frame, text="Sweep all 6 Pixel")
        frame.pack(padx=10, pady=10, fill="x", expand=True)
        #start voltage
        ttk.Label(frame, text="Start Voltage (mV):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.start_voltage_entry = ttk.Entry(frame, textvariable=self.start_voltage)
        self.start_voltage_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        #stop voltage
        ttk.Label(frame, text="Stop Voltage (mV):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.stop_voltage_entry = ttk.Entry(frame, textvariable=self.stop_voltage)
        self.stop_voltage_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        #sweep voltage
        ttk.Label(frame, text="Sweep Step (mV):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.sweep_step_entry = ttk.Entry(frame, textvariable=self.sweep_voltage)
        self.sweep_step_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        #sweep delay    
        ttk.Label(frame, text="Sweep Delay (msec):").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.sweep_delay_entry = ttk.Entry(frame, textvariable=self.sweep_delay)
        self.sweep_delay_entry.grid(row=3, column=1, padx=5, pady=5, sticky="ew")
        #area    
        ttk.Label(frame, text="Area (cm*cm):").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.area_entry = ttk.Entry(frame, textvariable=self.area_value)
        self.area_entry.grid(row=4, column=1, padx=5, pady=5, sticky="ew")
        #radiation    
        ttk.Label(frame, text="Radiation():").grid(row=5, column=0, padx=5, pady=5, sticky="w")
        self.radiation_entry = ttk.Entry(frame, textvariable=self.radiation_value)
        self.radiation_entry.grid(row=5, column=1, padx=5, pady=5, sticky="ew")
        #direction    
        ttk.Label(frame, text="Direction").grid(row=6, column=0, padx=5, pady=5, sticky="w")
        self.direction_combobox = ttk.Combobox(frame, textvariable=self.direction, values=["Up", "Down", "Both"])
        self.direction_combobox.grid(row=6, column=1, padx=5, pady=5, sticky="ew")
        #start sweep button
        self.start_sweep_button = ttk.Button(frame, text="Start Sweep", command=self.start_sweeping, state="disabled")
        self.start_sweep_button.grid(row=7, column=1, columnspan=2, pady=5, sticky="ew")

        # MPPT Testing section 
        frame = ttk.LabelFrame(self.scrollable_frame, text="MPPT Testing")
        frame.pack(padx=10, pady=10, fill="x", expand=True)

        ttk.Label(frame, text="Step voltage (mV):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.voltage_change_entry = ttk.Entry(frame, textvariable=self.voltage_change)
        self.voltage_change_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(frame, text="End voltage differ (mV):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.end_voltage_differ_entry = ttk.Entry(frame, textvariable=self.end_voltage_differ)
        self.end_voltage_differ_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        # Maximum Iterations
        ttk.Label(frame, text="Time Period (mins):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.time_period_entry = ttk.Entry(frame, textvariable=self.time_period)
        self.time_period_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        # Start MPPT Button
        self.start_mppt_button = ttk.Button(frame, text="Start MPPT", command=self.start_mppt)
        self.start_mppt_button.grid(row=3, column=1, columnspan=2, pady=5, sticky="ew")

        # Manual Control section 
        frame = ttk.LabelFrame(self.scrollable_frame, text="Sweep single pixel")
        frame.pack(padx=10, pady=10, fill="x", expand=True)

        # Pixel Number Selection
        ttk.Label(frame, text="Pixel Number:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.pixel_number_combobox = ttk.Combobox(frame, textvariable=self.pixel_number, values=[str(i) for i in range(6)])
        self.pixel_number_combobox.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # Number of Sweeps Input
        ttk.Label(frame, text="No. of sweeps:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.sweep_count_entry = ttk.Entry(frame, textvariable=self.sweep_count)
        self.sweep_count_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        # On Button
        self.on_button = ttk.Button(frame, text="On", command=self.turn_on_pixel)
        self.on_button.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        # Off Button
        self.off_button = ttk.Button(frame, text="Off", command=self.turn_off_pixel)
        self.off_button.grid(row=2, column=1, padx=5, pady=5, sticky="e")

        # Start Manual Sweep Button
        self.start_manual_sweep_button = ttk.Button(frame, text="Start Manual Sweep", command=self.start_manual_sweeping, state="disabled")
        self.start_manual_sweep_button.grid(row=3, column=1, columnspan=2, pady=5, sticky="ew")

    
    def get_com_ports(self):
        ports = serial.tools.list_ports.comports()
        connected_ports = [port.device for port in ports] #if port.device.startswith('COM')]
        return connected_ports
     
    def unlock_all(self):
        if connected:
            self.start_sweep_button.configure(state="normal")
            self.start_manual_sweep_button.configure(state="normal")

    def start_sweeping(self):
        output_folder = create_output_folder(self.user_name.get())
        output_file_path = mux(keithley, ser, 6, self.start_voltage.get(),self.stop_voltage.get(), self.sweep_voltage.get(), self.sweep_delay.get(), self.user_name.get(), self.area_value.get(), self.direction.get(), self.radiation_value.get(), output_folder)
        current_datetime = datetime.now()
        current_date = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        add_page_heading(output_file_path, "My Plots", self.user_name.get(), "Plots of pixels", current_date)

    def connect_keithley(self):
        print("Connecting to Keithley...")
        connect_keithley()
        if connected:
            print("Connected to Keithley!")
            self.connect_keithley_button.configure(state="disabled")
            self.disconnect_keithley_button.configure(state="normal")
        run_start_up_commands(keithley)

    def connect_multiplexer(self):
        k=self.com_port.get()
        serialopen(k)
        if ser.isOpen():
            print(f"Connecting to Multiplexer on port {k}...")
            self.unlock_all()
    
    def disconnect_keithley(self):
        try:
         keithley.close()
         connected=False
        except Exception as e:
          connected=True
        if not connected:
            print("Disconnecting Keithley...")
            self.connect_keithley_button.configure(state="normal")
            self.disconnect_keithley_button.configure(state="disabled")
            self.start_sweep_button.configure(state="disabled")
            self.start_manual_sweep_button.configure(state="disabled")
            ser.close()
        
    def turn_on_pixel(self):
        pixel_number = self.pixel_number.get()
        select_pixel(ser, pixel_number, turn_off=True)
        #self.on_button.configure(state="disabled")

    def turn_off_pixel(self):
        pixel_number = self.pixel_number.get()
        select_pixel(ser, pixel_number, turn_off=False)
        #self.on_button.configure(state="disabled")

    def start_manual_sweeping(self):
        output_folder = self.create_output_folder(self.user_name.get())
        output_file_path = manual_sweeping(keithley, ser, self.pixel_number.get(), self.sweep_count.get(), self.start_voltage.get(),self.stop_voltage.get(), self.sweep_voltage.get(), self.sweep_delay.get(), self.user_name.get(), self.area_value.get(), self.direction.get(), self.radiation_value.get(), output_folder)
        current_datetime = datetime.now()
        current_date = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        add_page_heading(output_file_path, "My Plots", self.user_name.get(), "Plots of pixels", current_date)

    def comval(self, event):
        selected_port = self.com_port.get()
        print(f"Selected COM port: {selected_port}")
    
    def start_mppt(self):
        output_folder = create_output_folder(self.user_name.get())
        best_pixel, Pmax, Pnom, Voc, Isc, Imax, Vmax, FF, efficiency = find_and_store_max_Pmax_pixel(self.area_value.get(), self.radiation_value.get())
        mppt(keithley, ser, best_pixel, Pmax, Pnom, Isc, Voc, Imax, Vmax, FF, efficiency, self.voltage_change.get(), self.end_voltage_differ.get(), self.time_period.get(), self.sweep_delay.get(), self.user_name.get(), self.radiation_value.get(), self.area_value.get(), output_folder)


if __name__ == "__main__":
    app = SolarCellTestApp()
    app.mainloop()

