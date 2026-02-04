import os
import cv2
import customtkinter as ctk

def ask_username():
    username = {"value": None}

    def submit():
        val = entry.get().strip()
        if val:
            username["value"] = val
            win.destroy()

    win = ctk.CTk()
    win.title("Register User")
    win.geometry("300x150")
    win.resizable(False, False)

    label = ctk.CTkLabel(win, text="Enter username")
    label.pack(pady=10)

    entry = ctk.CTkEntry(win)
    entry.pack(pady=5)

    btn = ctk.CTkButton(win, text="Register", command=submit)
    btn.pack(pady=10)

    win.mainloop()
    return username["value"]


def register_face(face_img, username):
    base_dir = "registered_faces"
    user_dir = os.path.join(base_dir, username)

    os.makedirs(user_dir, exist_ok=True)

    face_path = os.path.join(user_dir, "face.jpg")
    cv2.imwrite(face_path, face_img)

    return face_path
