import customtkinter as ctk
from PIL import Image
import os
import sys
from app.session_store import load_last_login

class WelcomeScreen(ctk.CTk):

    def __init__(self, username):
        super().__init__()

        self.username = username
        self.title("Welcome")
        self.geometry("700x500")
        self.resizable(False, False)

        self._build_ui()

    def _build_ui(self):

        # -------- PROFILE IMAGE --------
        img_path = os.path.join(
            "registered_faces", self.username, "profile.jpg"
        )

        if os.path.exists(img_path):
            img = Image.open(img_path)
        else:
            img = Image.new("RGB", (200, 200), color="gray")

        profile_img = ctk.CTkImage(
            light_image=img,
            dark_image=img,
            size=(180, 180)
        )

        ctk.CTkLabel(self, image=profile_img, text="").pack(pady=20)
        self.image_ref = profile_img  # prevent GC

        # -------- USER NAME --------
        ctk.CTkLabel(
            self,
            text=f"Welcome, {self.username} 👋",
            font=("Arial", 26, "bold")
        ).pack(pady=10)

        # -------- LAST LOGIN --------
        last_login = load_last_login(self.username)

        ctk.CTkLabel(
            self,
            text=f"Last Login: {last_login}",
            font=("Arial", 14)
        ).pack(pady=5)

        # -------- STATUS --------
        ctk.CTkLabel(
            self,
            text="Authentication Successful",
            font=("Arial", 16),
            text_color="#22C55E"
        ).pack(pady=15)

        # -------- LOGOUT --------
        ctk.CTkButton(
            self,
            text="Logout",
            width=160,
            command=self.logout
        ).pack(pady=25)

    def logout(self):
        self.destroy()
        os.execl(sys.executable, sys.executable, *sys.argv)
