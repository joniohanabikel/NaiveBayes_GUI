from tkinter import *
from tkinter import filedialog


class NaiveBayesGUI:
    def __init__(self, root):
        self.root = root

        self.folder = ""
        self.bins = 0

        self.root.title("Naive Bayes Classifier")
        self.general_frame = Frame(root, padx=100, pady=70)
        self.frame0 = Frame(self.general_frame,pady=10, padx=20)
        self.frame1 = Frame(self.general_frame,pady=10, padx=20)
        self.frame2 = Frame(self.general_frame,pady=10, padx=20)
        self.frame3 = Frame(self.general_frame,pady=10, padx=20)
        #grid
        # labels
        # 1. Directory Path
        self.path_label = Label(self.frame0, text="Directory Path")
        #build text entry
        self.path_entry = Entry(self.frame0, width=40, bg="white",highlightthickness=1)
        self.path_entry.config(highlightbackground="blue")
        #build Browse Button
        self.path_browse_btn = Button(self.frame0, text="Browse", padx=6, command=self.get_path)

        # 2. Discretization Bins
        valInput = root.register(self.get_bins)
        self.bins_user_input = IntVar()
        self.discretization_bins_label = Label(self.frame1, text="Discretization Bins:")
        self.bins_entry = Entry(self.frame1, width=10, bg="white", validate="key", validatecommand=(valInput))

        #3. Buttons
        self.build_button = Button(self.frame2, text="Build", padx=20, command=self.build_event)
        self.classify_button = Button(self.frame3, text="Classify", padx=20)


        #grid setup
        # self.root.geometry("600x300")
        self.general_frame.pack()
        self.frame0.grid(row=0)
        self.frame1.grid(row=1)
        self.frame2.grid(row=2)
        self.frame3.grid(row=3)
        self.path_label.grid(row=0, column=0, sticky=E)
        self.path_entry.grid(row=0, column=1, columnspan=2, sticky=E)
        self.path_browse_btn.grid(row=0, column=3, sticky=W+E)
        self.discretization_bins_label.grid(row=0, column=0, sticky=E)
        self.bins_entry.grid(row=0,column=1, columnspan=2, sticky=E)
        self.build_button.grid(row=3, column=1, sticky=W+E)
        self.classify_button.grid(row=4, column=1, sticky=W+E)


    def get_path(self):
        self.folder = filedialog.askdirectory(initialdir= "/", title='Choose a Working Directory')
        if self.folder:
            self.path_entry.insert(0, self.folder)


    def get_bins(self):
        number = self.bins_entry.get()
        if (self.bins == 0) | (not number):
            print("Select bins number")
        else:
            try:
                print(number)
                self.entered_number = int(number)
                return True
            except ValueError:
                return False



    def build_event(self):
        print(self.folder)
        print(self.bins_entry.get())


    def classfy_event(self):
        pass
if __name__ == '__main__':
    root = Tk()
    my_gui = NaiveBayesGUI(root)
    root.mainloop()
