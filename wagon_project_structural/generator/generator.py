import argparse
import itertools
import json
import subprocess
import sys
import uuid
from pathlib import Path
from subprocess import TimeoutExpired

import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point,LineString
from shapely.geometry.polygon import Polygon
import shapely.affinity as affinity




def random_quadrilateral(ratio_z_x=1, percent_x=0.1, percent_z=0.1, scale=1, trans=[0, 0]):
    """
    Permet de créer un quadrilatrère alétoire
    x----------------------------------x
    |                                  |
    |                                  |
    |                                  |
    |                                  |
    x----------------------------------x
    """


    assert ratio_z_x<=1

    ranges = [
        np.array([[0,0],
                  [percent_x,percent_z*ratio_z_x]]),
        np.array([[1-percent_x,0],
                  [1.0,percent_z*ratio_z_x]]),
        np.array([[1-percent_x,ratio_z_x-percent_z*ratio_z_x],
                  [1.0,ratio_z_x]]),
        np.array([[0,ratio_z_x-percent_z*ratio_z_x],
                  [percent_x,ratio_z_x]])
    ]

    points = []
    rng = np.random.default_rng()
    for range_ in ranges:
        point = rng.random((2,))
        for coord in range(2):
            point[coord] = trans[coord] + (range_[0,coord] + (range_[1,coord]-range_[0,coord])*point[coord])*scale
        points.append(point)
    return Polygon(points)

def plot_polygon(polygon,ax=None):
    points = np.array(list(polygon.exterior.coords))
    if ax is None:
        ax = plt.gca()
    ax.plot(points[:,0], points[:,1], 'r', lw=2)
    return ax


class Geometrie:
    def __init__(self,polygon):
        self.main_polygon = polygon
        self._internal_polygons = {}
        self._internal_points = {}
        self._internal_lines = {}
        bounds = np.array(polygon.bounds).reshape(2,2).transpose()
        self.dims = bounds[:,1]-bounds[:,0] # row 0 : x // row 1 : y
        self.min_dim = self.dims.min()
        self.max_dim = self.dims.max()
        self.set_material()


    def internal_lines(self,names=None):
        for name,lines in self._internal_lines.items():
            if names is None or name in names:
                for l in lines:
                    yield l

    def internal_points(self,names=None):
        for name,polys in self._internal_points.items():
            if names is None or name in names:
                for p in polys:
                    yield p

    def internal_polygons(self,names=None):
        for name,polys in self._internal_polygons.items():
            if names is None or name in names:
                for p in polys:
                    yield p

    def internal_point_names(self):
        return list(self._internal_points.keys())

    def internal_polygon_names(self):
        return list(self._internal_polygons.keys())

    def internal_lines_names(self):
        return list(self._internal_lines.keys())

    def reset_polygons(self,names=None):
        if names is None:
            names = self.internal_polygon_names()
        for name in names:
            self._internal_polygons.setdefault(name,[])
            self._internal_polygons[name] = []

    def reset_lines(self,names=None):
        if names is None:
            names = self.internal_lines_names()
        for name in names:
            self._internal_lines.setdefault(name,[])
            self._internal_lines[name] = []

    def reset_points(self,names=None):
        if names is None:
            names = self.internal_point_names()
        for name in names:
            self._internal_points.setdefault(name,[])
            self._internal_points[name] = []

    def add_tremie(self,ratio_z_x=1,ratio=0.1,max_iter=1000,ratio_distance_min=1,name="tremie"):
        rng = np.random.default_rng()
        if name not in self._internal_polygons:
            self._internal_polygons[name] = []
        dim = self.min_dim*ratio
        skip = True
        iter_ = 0
        while skip and iter_<=max_iter:
            skip = False
            iter_ += 1
            trans = rng.random((2,))
            tremie = random_quadrilateral(ratio_z_x=ratio_z_x, scale=ratio, trans=trans)
            if tremie.within(self.main_polygon):
                if tremie.distance(self.main_polygon.exterior)<=ratio_distance_min*dim:
                    skip = True
                if not skip:
                    for tr in self.internal_polygons():
                        if tremie.intersects(tr):
                            skip = True
                            break
                        if tremie.distance(tr.exterior)<=ratio_distance_min*dim:
                            skip = True
                            break
            else:
                skip = True
        if not skip:
            self._internal_polygons[name].append(tremie)
            return True
        return False

    def add_interior_point(self,center_area=None,ratio_area=1,ratio_distance_min=0.001,max_iter=1000,name="point"):
        rng = np.random.default_rng()
        if name not in self._internal_points:
            self._internal_points[name] = []
        dim = self.max_dim*ratio_area
        skip = True
        iter_ = 0
        while skip and iter_<=max_iter:
            skip = False
            iter_ += 1
            point = rng.random((2,))*dim
            if center_area:
                point = point + np.array(center_area)
            point_ = Point(point)
            if point_.within(self.main_polygon):
                if point_.distance(self.main_polygon.exterior)<=ratio_distance_min*dim:
                    skip = True
                if not skip:
                    for tr in self.internal_polygons():
                        if point_.within(tr):
                            skip = True
                            break
                        if point_.distance(tr.exterior)<=ratio_distance_min*dim:
                            skip = True
                            break
            else:
                skip = True
            if not skip:
                self._internal_points[name].append(point_)
                return True
        return False


    def add_interior_line(self, from_line=None, ratio_length=0.8, ratio_distance_min=0.1, range_angle=5, max_iter=1000, name="line"):
        rng = np.random.default_rng()
        if name not in self._internal_lines:
            self._internal_lines[name] = []
        if from_line is None:
            from_line = rng.integers(4)


        if from_line%2==0:
            coefs = np.array([0,1])
        else:
            coefs = np.array([1,0])
        if from_line>1:
            coefs = -coefs

        coords = self.main_polygon.exterior.coords

        skip = True
        iter_ = 0

        while skip and iter_<=max_iter:
            skip = False
            iter_ += 1
            l  = LineString((Point(coords[from_line%4]),Point(coords[(from_line+1)%4])))
            dim = ratio_distance_min*l.length

            l=affinity.scale(l, xfact=ratio_length, origin="centroid")

            angle = -range_angle/2+range_angle*rng.random((1,))
            off = dim + np.abs(l.length/2*np.tan(np.radians(angle)))

            l=affinity.rotate(l,angle=angle,origin="centroid",use_radians=False)
            l=affinity.translate(l,xoff=coefs[0]*off,yoff=coefs[1]*off)

            if l.within(self.main_polygon):
                # if l.distance(self.main_polygon.exterior)<=dim:
                #     skip = True
                if not skip:
                    for tr in self.internal_polygons():
                        if l.within(tr):
                            skip = True
                            break
                        if l.distance(tr.exterior)<=ratio_distance_min*dim:
                            skip = True
                            break
            else:
                skip = True
            if not skip:
                self._internal_lines[name].append(l)
                return True
        return False

    def plot(self,ax=None):
        ax = plot_polygon(self.main_polygon,ax=ax)
        for name,polys in self._internal_polygons.items():
            for p in polys:
                ax = plot_polygon(p)

        for name,pts in self._internal_points.items():
            coords = []
            for pt in pts:
                coords.append(pt.coords[0])
            coords = np.array(coords)
            ax.scatter(coords[:,0],coords[:,1],label=name,marker="x")
        return ax

    def tm(self):
        results = []
        for tr in self.internal_polygons():
            results.append(tr.bounds)
        if results:
            results = np.vstack(results)
            return (results[:,2:]-results[:,:2]).min()/2
        return self.min_dim/20

    @staticmethod
    def polygon_to_pyth(no,polygon,nos = None,td=None):
        if nos is None:
            no_end = no+400
            nos = np.arange(no,no_end,100)
        else:
            no_end = no
            for i,no in enumerate(nos):
                if no is None:
                    no_end+=100
                    nos[i] = no_end
        s_no = ' '.join(map(str,nos))
        s_td = ""
        coords = np.array(polygon.exterior.coords[:-1])
        s_x = ', '.join(map(str,coords[:,0]))
        s_z = ', '.join(map(str,coords[:,1]))

        if td is not None:
            s_td = f"TD {td}"
        s = f"""\
NO {s_no} XX {s_x} ZZ {s_z} YY 0
CF {no} NO {s_no} {s_td}
"""
        return no_end+100,s

    def set_material(self,ep=0.2,mo=11e6,nu=0):
        self.ep = 0.2
        self.mo = 11e6
        self.nu = 0


    def to_don(self):
        no = 100
        s=f"""\
** GEO TM {self.tm()}
TD 1 EP {self.ep} MO {self.mo} NU {self.nu}
"""
        # CONTOURS EXTERIEURS
        cf_ext_num = no
        no,delt_s = self.polygon_to_pyth(no,self.main_polygon)
        s+=delt_s
        cf_int_nums = []

        # CONTOURS INTERIEURS
        for tr in self.internal_polygons(("tremie",)):
            cf_int_nums.append(no)
            no,delt_s = self.polygon_to_pyth(no,tr)
            s+=delt_s

        # NO_APPUIS
        coords = []
        for pt in self.internal_points(names=("appui",)):
            coords.append(pt.coords[0])

        s_no_app_int_in = ""
        s_no_app_int = ""
        if coords:
            coords = np.array(coords)
            no_end = no + coords.shape[0]*100
            no_app_int = np.arange(no,no_end,100)
            s_no_app_int_in = "IN "
            s_no_app_int = " ".join(map(str,no_app_int))
            s_x = ", ".join(map(str,coords[:,0]))
            s_z = ", ".join(map(str,coords[:,1]))
            s+=f"""NO {s_no_app_int} XX {s_x} ZZ {s_z} YY 0
    """

        # NO_LINE
        coords = []
        for l in self.internal_lines(names=("appui",)):
            coords.append(l.coords[0])
            coords.append(l.coords[1])

        s_li_app_int_il = ""
        s_li_app_int= ""
        if coords:
            coords = np.array(coords)
            no_end = no + coords.shape[0]*100
            no_app_int = np.arange(no,no_end,100)
            s_li_app_int_il = "IL "
            s_no_app_li_int = " ".join(map(str,no_app_int))
            s_li_app_int = " ".join(map(str,no_app_int[::2,]))
            s_x = ", ".join(map(str,coords[:,0]))
            s_z = ", ".join(map(str,coords[:,1]))
            s_no2 = " ".join(map(str,no_app_int[3::-2,][::-1,]))
            s+=f"""NO {s_no_app_li_int} XX {s_x} ZZ {s_z} YY 0
LI {s_li_app_int} SE N1 {s_li_app_int} N2 {s_no2}
"""

        # SURFACE
        s += f"""\
SU {cf_ext_num} PL CF {cf_ext_num} {' '.join(map(str,cf_int_nums))} {s_no_app_int_in}{s_no_app_int} {s_li_app_int_il}{s_li_app_int} TD 1
"""
        # APPUIS
        s+=f"""
** ZONE
"""
        if s_no_app_int:
            s+=f"""NO {s_no_app_int} APP01
"""
        if s_li_app_int:
            s+=f"""NO LI {s_li_app_int} APP01
"""

        
        s+= f"""** APPUIS
NO ZO APP01 YY 1e10 XX 1e-3 ZZ 1e-3
"""
        s += "** FIN"
        return s

    def to_load(self):
        coords = []
        for pt in self.internal_points(names=("force",)):
            coords.append(pt.coords[0])
        coords = np.array(coords)
        s_f_x = ",".join(map(str,coords[:,0]))
        s_f_y = ",".join(map(str,coords[:,1]))
        s =f"""\
** APP DG
** ACT TO
** CAS FIC 100
LIB PO XX {s_f_x} YY {s_f_y} GL FY -10
** FIN"""
        return s

    def to_list(self):
        s = """\
** IMP FIC 100
RA
** FIN"""
        return s

    def to_list_points(self):
        s = r"""** VAR GEO
$o 1 "OUTPUT\slist_points.csv"
$E FIC 1 NO;XX;YY;ZZ
$B $.NO = NO ZO APP01
$.XX = XX($.NO)
$.YY = YY($.NO)
$.ZZ = ZZ($.NO)
$E FIC 1 $.NO ; $.XX ; $.YY ; $.ZZ
$F
$c 1
** FIN
"""
        return s


class Generator:
    def __init__(self,params,pyth_path=None):
        self.params = params
        self.pyth_path = Path(pyth_path)
        if self.pyth_path.is_file():
            self.pyth_path = self.pyth_path.parent
        self.pyth_path_ini = self.pyth_path / "pythagore.ini"

    def main(self,n,cwd=Path.cwd(),plot_each=100):
        ratios = self.params.get("main_polygon_ratio_z_x",[1])
        percent_x = self.params.get("main_polygon_percent_x",[0.1])
        percent_z = self.params.get("main_polygon_percent_z",[0.1])
        n_tremies =  self.params.get("tremies_n",[0.1])
        n_appuis =  self.params.get("appuis_n",[4])
        n_forces =  self.params.get("forces_n",[1])
        nconfs = len(ratios)*len(percent_x)*len(percent_z)
        nforces_ = len(n_forces)
        ntremies_ = len(n_tremies)
        nappuis_ = len(n_appuis)
        for ig in range(n):
            num = -1
            for iconf,(r,px,pz) in enumerate(itertools.product(ratios,percent_x,percent_z)):
                print(f"r:{r} px:{px} pz:{pz}")
                self.geom = Geometrie(random_quadrilateral(
                    ratio_z_x=r,
                    percent_x=px,
                    percent_z=pz,
                ))

                for i_tr,nt in enumerate(n_tremies):
                    self.geom.reset_polygons()
                    self.geom.reset_points()
                    self.add_tremies(nt)
                    for i_app,na in enumerate(n_appuis):
                        self.add_appuis(na)
                        print(f"...maillage geom {ig+1}/{n} conf {iconf+1}/{nconfs} tremie {i_tr+1}/{ntremies_} appuis {i_app+1}/{nappuis_}",)
                        if self.mailler(cwd):
                            for iforce,nf in enumerate(n_forces):
                                num +=1
                                print(f"...geom {ig+1}/{n} {iconf+1}/{nconfs} {i_tr+1}/{ntremies_} {i_app+1}/{nappuis_} force {iforce+1}/{nforces_}",)
                                self.add_forces(nf)
                                sub = self.subfolder_name(nt,na,nf)
                                # TODO WRITE LOAD
                                if self.load(cwd=cwd):
                                    print(f"...geom {ig+1}/{n} {iconf+1}/{nconfs} {i_tr+1}/{ntremies_} {i_app+1}/{nappuis_} post {iforce+1}/{nforces_}",)
                                    if self.posttraite(cwd):
                                        self.store(sub,cwd=cwd,plot=num%plot_each==0)
                        else:
                            print("...... fail")


    def subfolder_name(self,nt,na,nf):
        return f"{nt}tr_{na}ap_{nf}fo"

    def add_tremies(self,nt):
        self.geom.reset_polygons(names=("tremie",))
        for _ in range(nt):
            self.geom.add_tremie(name="tremie")

    def add_appuis(self,na):
        self.geom.reset_points(names=("appui",))
        for _ in range(na):
            self.geom.add_interior_point(name="appui")

    def add_forces(self,nf):
        self.geom.reset_points(names=("force",))
        for __ in range(nf):
            self.geom.add_interior_point(name="force")

    @staticmethod
    def to_decoda_string(s):
        return "\n".join(map(Generator.split_decoda_line,s.splitlines()))

    @staticmethod
    def split_decoda_line(line):
        if len(line)>60:
            i = 59
            while i>0:
                if line[i]==" ":
                    return line[:i] + " S\n" + Generator.split_decoda_line(line[i+1:])
                i-=1
        else:
            return line

    def mailler(self,cwd=Path.cwd()):
        self.points_results = None
        with open(cwd / "DDON", 'w') as f:
            f.write(Generator.to_decoda_string(self.geom.to_don()))
        with open(cwd / "DLIST", 'w') as f:
            f.write(Generator.to_decoda_string(self.geom.to_list_points()))
        with open(cwd / "DMODELE", 'w') as f:
            f.write(Generator.to_decoda_string("""** EXE
DDON
DLIST
** FIN
"""))
        res = self.common_launch("modele","DMODELE",cwd)
        if res:
            plist =cwd / "OUTPUT" / "slist_points.csv"
            if plist.exists():
                df = pd.read_csv(plist,sep=";",dtype=float,header=0,names=["NO","XX","YY","ZZ"]).sort_values(by="NO")
                self.points_results = df.to_dict()
            else:
                res = False
            print("...... list")
        return res



    def common_launch(self,module_name,file_name,cwd=Path.cwd(),print_out=False):
        try:
            compproc = subprocess.run([self.pyth_path / "CORE" / f"{module_name}.exe",self.pyth_path_ini,file_name],
                                  capture_output=True, encoding="cp1252", cwd=cwd,
                                  )
        except TimeoutExpired:
            # popen.kill()
            # out,err = popen.communicate()
            return False
        if print_out:
            print(compproc.stdout)
        return "E*" not in compproc.stdout and compproc.returncode==0

    def load(self,cwd=Path.cwd()):
        self.uuid = uuid.uuid4()

        self.results = None

        with open(cwd / "DLOAD", 'w') as f:
            f.write(Generator.to_decoda_string(self.geom.to_load()))
        with open(cwd / "DLIST", 'w') as f:
            f.write(Generator.to_decoda_string(self.geom.to_list()))


        plist = cwd / "OUTPUT" / 'slist.csv'
        if plist.exists():
            plist.unlink()
        res = self.common_launch("load","DLOAD",cwd)
        print("...... load")
        if res:
            self.common_launch("list","DLIST",cwd)
            print("...... list")
        return res

    def posttraite(self,cwd=Path.cwd()):
        plist =cwd / "OUTPUT" / "slist.csv"
        if plist.exists():
            try:
                df = pd.read_csv(plist,sep=";",skiprows=5).sort_values(by="NO")
                self.results = list(df.RY.values)
                return True
            except:
                return False



    KEY_MAIN_POLYGON = 0
    KEY_POLYS = 1
    KEY_POINTS = 2
    KEY_RESULTS = 3
    KEY_POINTS_RESULTS = 4

    def store(self,subfolder="main",cwd=Path.cwd(),plot=True):

        obj = {}
        obj[self.KEY_MAIN_POLYGON] = np.array(self.geom.main_polygon.exterior.coords[:-1]).tolist()
        obj[self.KEY_POLYS] = {}
        for name,polys in self.geom._internal_polygons.items():
            obj[self.KEY_POLYS][name] = []
            for p in polys:
                obj[self.KEY_POLYS][name].append(np.array(p.exterior.coords[-1]).tolist())
        obj[self.KEY_POINTS] = {}
        for name,pts in self.geom._internal_points.items():
            obj[self.KEY_POINTS][name] = []
            for p in pts:
                obj[self.KEY_POINTS][name].append(np.array(p.coords[0]).tolist())
        obj[self.KEY_POINTS_RESULTS] = self.points_results
        obj[self.KEY_RESULTS] = self.results
        p = cwd / "results" / subfolder
        p.mkdir(exist_ok=True,parents=True)
        if plot:
            try:
                fig = self.plot()
                plt.savefig(p / f"{self.uuid}.png")
                plt.close(plt.gcf())
                print("...... plot")
            except:
                pass
        with open(p / f"{self.uuid}.json","w") as f:
            json.dump(obj,f)

    @staticmethod
    def load_subfolder(subfolder,cwd=Path.cwd()):
        X,y = [],[]
        for p in cwd.iterdir():
            if p.is_file() and p.suffix==".json":
                with open(cwd / "results" / subfolder / "{self.uuid}.json","w") as f:
                    obj = json.load(f)
                y.append([p.name,]+obj[Geometrie.KEY_RESULTS])
                # TODO TRAITE KEY_POINTS_RESULTS
                X.append(obj)
        return X,y

    def plot(self):
        plt.figure()
        ax = self.geom.plot()
        return ax


class GeneratorOneForceAlignedSupports(Generator):
    def add_appuis(self,na):
        self.geom.reset_lines(names=("appui",))
        # na est le ratio
        r = 1/na
        self.geom.add_interior_line(0,name="appui",ratio_distance_min=r)
        self.geom.add_interior_line(2,name="appui",ratio_distance_min=r)

    def plot(self):
        ax = super(GeneratorOneForceAlignedSupports, self).plot()
        coords = pd.DataFrame(self.points_results)[["XX","ZZ"]].values
        ax.scatter(coords[:,0],coords[:,1],label="app_li",marker="x")
        return ax

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pythagore_path",
                        default=r"R:\INFORMATIQUE\APPLICATIONS\TPI\Pythagore\version_21.02\Pythagore.exe")
    parser.add_argument("--cwd",
                        default=None)
    parser.add_argument("--n",
                        default=100,type=int)
    parser.add_argument("--plot_each",
                        default=100,type=int)
    args = parser.parse_args(sys.argv[1:])

    cwd = args.cwd
    if cwd is None:
        cwd = Path.cwd()
    else:
        cwd = Path(cwd)

    # PARAMS
    params_path = cwd / "params.json"
    assert params_path.exists(),f"Chemin vers params {params_path} non reconnu"
    with open(params_path,"r") as f:
        params = json.load(f)

    gen_class_name = params.pop("gen_class","GeneratorOneForceAlignedSupports")
    gen_class = globals().get(gen_class_name)
    assert gen_class is not None
    gen = gen_class(
        params,
        pyth_path=args.pythagore_path
    )
    gen.main(args.n,cwd=cwd,plot_each=args.plot_each)

if __name__ == '__main__':
    main()
