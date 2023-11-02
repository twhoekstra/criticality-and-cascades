# In our environment manim is not installed. To run this code you will need to manually install manim and latex.
from manim import *

class Titlescreen(Scene):
    def construct(self):
        text1 = Tex("Definition Criticality:", font_size=60)

class Scene05(Scene):
    def construct(self):
        text1 = Tex("What is Criticality?", font_size=60)

        text2 = Tex("Definition Criticality:", font_size=60)
        text3 = Tex("The state in which your brain operates near a \\underline{phase transition} point between \\underline{order} and \\underline{chaos} .", 
                    font_size=40)
        
        text3.next_to(text2, DOWN, buff=0.5)

        self.wait()
        self.play(Write(text1))
        self.play(text1.animate.shift(UP))
        self.play(Write(text2))
        self.play(Write(text3), run_time = 5)
        self.wait(2)
        self.remove(text1)
        self.remove(text2)
        self.remove(text3)

class Scene1(Scene):
    def construct(self):
        text1 = Tex("Up and down states", font_size=60)

        up_down = ImageMobject("Up_down_state.png")

        text1.next_to(up_down, UP)

        self.wait()
        self.play(Write(text1))
        self.wait()
        self.add(up_down)
        self.wait()

class Scene2(Scene):
    def construct(self):
        # Create an up arrow in blue
        text1 = Tex("Ising model", font_size = 60)
        up_arrow = Arrow(color=BLUE)
        up_arrow.rotate(PI/2)

        text1.next_to(up_arrow, UP)
        self.wait()
        self.play(Write(text1))
        self.play(Create(up_arrow))
        self.wait(1)

        # Rotate the arrow to point down (180 degrees)
        down_arrow = up_arrow.copy()
        down_arrow.rotate(PI)
        down_arrow.set_color(RED)

        self.play(Transform(up_arrow, down_arrow))
        self.wait(1)

