# import the library
from appJar import gui

# handle button events
def press(button):
    if button == "Cancel":
        app.stop()
    else:
        usr = app.getEntry("Username")
        pwd = app.getEntry("Password")
        print("User:", usr, "Pass:", pwd)
        if button == "Submit":
            app.showSubWindow("Preferences")

# create a GUI variable called app
app = gui("Login Window", "800x400")
app.setBg("green")
app.setFont(18)

# add & configure widgets - widgets get a name, to help referencing them later
app.addLabel("title", "BookTender LogIn")
app.setLabelBg("title", "white")
app.setLabelFg("title", "green")

app.addLabelEntry("Username")
app.addLabelSecretEntry("Password")

# link the buttons to the function called press
app.addButtons(["Submit", "Cancel"], press)

app.setFocus("Username")

# Status Bar
app.addStatusbar(fields=3)
app.setStatusbar("CmpE 131", 0)
app.setStatusbar("Software Eng", 1)
app.setStatusbar("Group 6", 2)

#Subwindow
app.startSubWindow("Preferences", "400x200")
app.setBg("white")
app.setFont(18)
app.addLabel("title2", "Genres")
app.addCheckBox("All")
app.addCheckBox("Fantasy")
app.addCheckBox("Mystery")
app.addCheckBox("Fiction")
app.setCheckBox("All")

app.addLabel("title3", "Age Group")
app.addLabelOptionBox("Options", ["Kindergarden", "Elementary", "Teen", "Young Adult", "College"])

app.addButtons(["Set"], press)
app.stopSubWindow()

# start the GUI
app.go()