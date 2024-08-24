import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog
from impedancefitter import Fitter, available_file_format

directory = None


def browse_button():
    global directory
    directory = filedialog.askdirectory()


class MainWindow(tk.Frame):

    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.master = master
        self.init_window()

    # Creation of init_window
    def init_window(self):

        self.master.title("Impedancefitter")

        self.pack(fill=tk.BOTH, expand=1)

        menu = tk.Menu(self.master)
        self.master.config(menu=menu)

        file = tk.Menu(menu)
        file.add_command(label="Quit", command=self.client_exit)

        menu.add_cascade(label="File", menu=file)

        edit = tk.Menu(menu)

        edit.add_command(label="Initialize fitter", command=self.launch_fitter)

        menu.add_cascade(label="Fitting", menu=edit)

    def client_exit(self):
        answer = messagebox.askyesno(title='Quit',
                                     message='Are you sure you want to quit?')
        if answer:
            exit()

    def launch_fitter(self):
        dialog = FitterInit(self)
        self.fitter = dialog.result
        if isinstance(self.fitter, Fitter):
            print(self.fitter.inputformat)
        else:
            messagebox.showwarning(message="The fitter could not be initialized!")


class FitterInit(simpledialog.Dialog):

    def body(self, master):
        self.filetype = tk.StringVar()
        self.fitterkwargs = {}
        # use only warning for GUI
        self.fitterkwargs['LogLevel'] = 'WARNING'

        formats = available_file_format()
        self.filetype.set(formats[0])
        option = tk.OptionMenu(master, self.filetype, *formats)
        option.grid(row=0, columnspan=2, sticky=tk.W)

        m = tk.Label(master, text="Optional variables below. Can be left empty.")
        m.grid(row=1, columnspan=2, sticky=tk.W)

        tk.Label(master, text="Minimum frequency:").grid(row=2, sticky=tk.W)
        tk.Label(master, text="Maximum frequency:").grid(row=3, sticky=tk.W)
        tk.Label(master, text="File directory:").grid(row=4, sticky=tk.W)

        valfloat = master.register(self.validate_float)
        vcmd = (valfloat, '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
        self.e1 = tk.Entry(master, validate="key",
                           validatecommand=vcmd)
        self.e2 = tk.Entry(master, validate="key",
                           validatecommand=vcmd)

        self.e1.grid(row=2, column=1)
        self.e2.grid(row=3, column=1)
        self.e3 = tk.Button(master, text="Browse", command=browse_button).grid(row=4, column=1)
        """
        if 'ending' in kwargs:
            self.ending = kwargs['ending']
        if 'data_sets' in kwargs:
            self.data_sets = kwargs['data_sets']
        if 'current_threshold' in kwargs:
            self.current_threshold = kwargs['current_threshold']
        if 'write_output' in kwargs:
            self.write_output = kwargs['write_output']
        if 'fileList' in kwargs:
            self.fileList = kwargs['fileList']
        if 'trace_b' in kwargs:
            self.trace_b = kwargs['trace_b']
        if 'skiprows_txt' in kwargs:
            self.skiprows_txt = kwargs['skiprows_txt']
        if 'skiprows_trace' in kwargs:
            self.skiprows_trace = kwargs['skiprows_trace']
        if 'show' in kwargs:
            self.show = kwargs['show']
        if 'savefig' in kwargs:
            self.savefig = kwargs['savefig']
        if 'delimiter' in kwargs:
            self.delimiter = kwargs['delimiter']
        """

    def apply(self):
        self.result = Fitter(self.filetype.get(), **self.fitterkwargs)

        if len(self.e1.get()) > 0:
            self.fitterkwargs['minimumFrequency'] = float(self.e1.get())
        if len(self.e2.get()) > 0:
            self.fitterkwargs['maximumFrequency'] = float(self.e2.get())
        self.fitterkwargs['directory'] = directory
        self.validate_kwargs()

    def validate_float(self, d, i, P, s, S, v, V, W):
        if P:
            # allow decimal numbers
            if P[-1] == 'e' and P.count('e') == 1:
                return True
            try:
                float(P)
                return True
            except ValueError:
                return False
        else:
            return True

    def validate_kwargs(self):
        if 'maximumFrequency' in self.fitterkwargs and 'minimumFrequency' in self.fitterkwargs:
            if self.fitterkwargs['maximumFrequency'] < self.fitterkwargs['minimumFrequency']:
                messagebox.showwarning(message="Maximum frequency must be greater than minimum frequency. Please repeat fitter initialization")
                self.result = None
            else:
                return
        return


if __name__ == "__main__":

    root = tk.Tk()

    # size of the window
    root.geometry("800x600")

    app = MainWindow(master=root)
    root.mainloop()
