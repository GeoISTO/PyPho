import pypho
from pypho import graphics
obj, cam, viewer = pypho.graphics.get_default_scene(show= False)
viewer.set_active_scalars("resolution")
viewer.show()
