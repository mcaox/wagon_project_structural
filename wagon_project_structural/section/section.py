from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

class Section(ABC):
    AREA_KEY = "area"
    MSTAT_AINF_KEY = "mstat_ainf"
    VP_KEY = "vp"
    INERTIA_KEY = "inertia"

    def __init__(self,nbpas=1e4):
        self.cache = {}
        self.nbpas = int(nbpas)
        self._nattrs = 0

    def print_ecart(self, f, a, b):
        ecart = (a - b) / b
        alerte = "! <------------------------------- ECART" if ecart > 0.05 else ""
        print(f, a, b, ecart, alerte)

    def verif(self):
        print("\n---------------- ")
        for f in [self.INERTIA_KEY, self.AREA_KEY, self.VP_KEY, self.MSTAT_AINF_KEY]:
            a = getattr(self, f)()
            b = getattr(self, f + "_calc")()
            self.print_ecart(f, a, b)

    @abstractmethod
    def h(self):
        return

    @abstractmethod
    def b(self, z):
        """

        :param z: par rapport à a_inf
        :return:
        """
        return

    def plot(self, n = 100,ax=None):
        zs = np.linspace(0,self.h(),n)
        bs = np.fromiter(map(self.b,zs),float)/2
        zs = np.append(zs,zs[::-1])
        bs = np.append(bs,-bs[::-1])
        zs = np.append(zs,zs[0])
        bs = np.append(bs,bs[0])
        if ax is None:
            ax = plt.gca()
            ax.set_aspect(1)
        ax.plot(bs, zs)
        return ax


    def integre(self, formule, cache=None, zmnmx=tuple()):
        calc = cache is None
        if not calc:
            cache_ = cache + "_".join(map(str, zmnmx))
            calc = cache_ not in self.cache.keys()
            if not calc:
                res = self.cache[cache_]
        if calc:
            if zmnmx:
                zs = np.linspace(zmnmx[0],zmnmx[1],self.nbpas)
                dz = (zmnmx[1] - zmnmx[0]) / self.nbpas
            else:
                zs = np.linspace(0,self.h(),self.nbpas)
                dz = self.h() / self.nbpas
            f_at_z = np.fromiter(map(formule,zs),float)
            res = f_at_z.sum()*dz

            if cache is not None:
                self.cache[cache_] = res
        return res


    def summary(self):
        return {
            self.AREA_KEY:self.area(),
            self.INERTIA_KEY:self.inertia(),
            self.VP_KEY:self.vp(),
            self.MSTAT_AINF_KEY:self.mstat_ainf(),
        }

    def area_calc(self):
        return self.integre(self.b, cache=self.AREA_KEY)

    def mstat_ainf_calc(self):
        return self.integre(lambda z: z * self.b(z), cache=self.MSTAT_AINF_KEY)

    def vp_calc(self):
        return self.mstat_ainf_calc() / self.area_calc()

    def inertia_calc(self, delta=0):
        return self.integre(lambda z: z * z * self.b(z), cache=self.INERTIA_KEY) + (
                delta ** 2 - self.vp_calc() ** 2) * self.area_calc()

    def area(self):
        return self.area_calc()

    def mstat_ainf(self):
        return self.mstat_ainf_calc()

    def vp(self):
        return self.vp_calc()

    def inertia(self, delta=0):
        return self.inertia_calc(delta=delta)

    def v(self):
        return self.h() - self.vp()


class Rect(Section):
    def __init__(self, b=0.2, h=0.8,**kwargs):
        super().__init__()
        self._b = b
        self._h = h
        self._nattrs = 2

    def h(self):
        return self._h

    def b(self, z):
        return self._b

class ISect(Section):
    def __init__(self, b=0.6, h=0.8,e=0.03,t=0.01):
        super().__init__()
        self._b = b
        self._h = h
        self._e = e
        self._t = t
        self._nattrs = 4

    def h(self):
        return self._h

    def b(self, z):
        if z<0:
            return 0
        if z<=self._e:
            return self._b
        if z<=self.h()-self._e:
            return self._t
        if z<=self.h():
            return self._b
        raise Exception

if __name__ == '__main__':
    sect = ISect(0.5,2,0.05,0.02)
    fig,axes = plt.subplots()
    sect.plot(ax=fig.axes[0])
    fig.savefig("test.png")
    fig.show()
    pass

