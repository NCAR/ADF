"""
Website (web) generation class for the
Atmospheric Diagnostics Framework (ADF).
This class inherits from the AdfObs class.

Currently this class does three things:

1.  Initializes an instance of AdfObs.

2.  Determines if a website will be generated.

3.  Sets website-related internal ADF variables.

This class also provides a method for generating
a website, as well as a method to add an image
file or pandas dataframe to the website.
"""

#++++++++++++++++++++++++++++++
#Import standard python modules
#++++++++++++++++++++++++++++++

import os
import os.path

from pathlib import Path

#+++++++++++++++++++++++++++++++++++++++++++++++++
#import non-standard python modules, including ADF
#+++++++++++++++++++++++++++++++++++++++++++++++++

#ADF modules:
from adf_obs import AdfObs

#Try to import Pandas.  If not available
#then simply notify website generator that
#no pandas dataframes will be passed in.
_PANDAS_DF = True
try:
    import pandas as pd
except ImportError:
    _PANDAS_DF = False
#End except

#++++++++++++++++++++++++++++
#Define web data helper class
#++++++++++++++++++++++++++++

class _WebData:

    """
    Class that stores all of
    the data and metadata from
    the "add_website_data" method
    needed by the website generator.
    """

    def __init__(self, web_data, web_name, case_name,
                 plot_ext = None,
                 category = None,
                 season = None,
                 plot_type = "Special",
                 data_frame = False,
                 html_file  = None,
                 asset_path = None,
                 multi_case = False):

        #Initialize relevant website variables:
        self.name       = web_name
        self.data       = web_data
        self.case       = case_name
        self.category   = category
        self.season     = season
        self.plot_type  = plot_type
        self.plot_ext   = plot_ext
        self.data_frame = data_frame
        self.html_file  = html_file
        self.asset_path = asset_path
        self.multi_case = multi_case

#+++++++++++++++++++++
#Define main web class
#+++++++++++++++++++++

class AdfWeb(AdfObs):

    """
    Website class, which initializes
    an AdfObs object and provides
    additional variables and methods
    needed for website generation.
    """

    def __init__(self, config_file, debug=False):

        """
        Initalize ADF Web object.
        """

        #Initialize Obs attributes:
        super().__init__(config_file, debug=debug)

        #Initialize website mean plots dictionary:
        self.__website_data = []

        #Initialize website plot type order lists:
        self.__plot_type_order = []

        #Initialize another plot type list for multi-case plots:
        self.__plot_type_multi = []

        #Initialize website plot type

        #Set case website path dictionary:
        #--------------------------------
        self.__case_web_paths = {}

        #Extract needed variables from yaml file:
        case_names = self.get_cam_info('cam_case_name', required=True)

        #Also extract baseline case (if applicable), and append to case_names list:
        if not self.compare_obs:
            baseline_name = self.get_baseline_info('cam_case_name', required=True)
            #Append baseline to case list:
            case_names.append(baseline_name)
        #End if

        #Loop over model cases and generate relevant website directories:
        for case_idx, case_name in enumerate(case_names):

            #Create new path object from user-specified plot directory path:
            plot_path = Path(self.plot_location[case_idx])

            #Create directory path where the website will be built:
            website_dir = plot_path / "website"

            #Create a directory path that will hold just the html files for individual images:
            img_pages_dir = website_dir / "html_img"

            #Create a directory path that will hold copies of the actual images:
            assets_dir = website_dir / "assets"

            #Create a directory that will hold table html files:
            table_pages_dir = website_dir / "html_table"

            #Specify where CSS files will be stored:
            css_files_dir = website_dir / "templates"

            #Add all relevant paths to dictionary for specific case:
            self.__case_web_paths[case_name] = {'website_dir': website_dir,
                                                'img_pages_dir': img_pages_dir,
                                                'assets_dir': assets_dir,
                                                'table_pages_dir': table_pages_dir,
                                                'css_files_dir': css_files_dir}
        #End for
        #--------------------------------

        #Finally, if this is a multi-case run, then add a "multi-case" option as well:
        if self.num_cases > 1:
            website_dir     = Path(self.get_basic_info('cam_diag_plot_loc', required=True))
            website_dir     = website_dir / "main_website"
            img_pages_dir   = website_dir / "html_img"
            assets_dir      = website_dir / "assets"
            table_pages_dir = website_dir / "html_table"
            css_files_dir = website_dir / "templates"

            self.__case_web_paths['multi-case'] = {'website_dir': website_dir,
                                                   'img_pages_dir': img_pages_dir,
                                                   'assets_dir': assets_dir,
                                                   'table_pages_dir': table_pages_dir,
                                                   'css_files_dir': css_files_dir}
        #End if

    #########

    # Create property needed to return "create_html" logical to user:
    @property
    def create_html(self):
        """Return the "create_html" logical to user if requested."""
        return self.get_basic_info('create_html')

    #########

    def add_website_data(self, web_data, web_name, case_name,
                         plot_ext = None,
                         category = None,
                         season = None,
                         plot_type = "Special",
                         multi_case=False):

        """
        Method that provides scripts a way to add an image file or
        Pandas dataframe to the website generator.

        Required Inputs:

        web_data  ->  Either a path to an image file, or a pandas dataframe.
        web_name  ->  The name of the plot or table (usually the plotted variable or case name).
        case_name ->  The name of the model case or dataset associated with the plot or table.

        Optional Inputs:

        category   -> Category for associated variable.  If not provided then generator will
                      attempt to grab it from the variable defaults file.  If no default is present
                      then it will default to "No category yet".
        season     -> What the season is for the plot.  If not provided it will assume the
                      plot does not need any seasonal seperation.
        plot_type  -> Type of plot.  If not provided then plot type will be "Special".

        multi_case -> Logical which indicates whether the image or dataframe can contain
                      multiple cases (e.g. a line plot with one line for each case).

        """

        #Do nothing if user is not requesting a website to be generated:
        if not self.create_html:
            return
        #End if

        #Initialize Pandas data frame logical:
        data_frame = False

        html_file = []
        #Check that the web_data is either a path
        #or a pandas dataframe:
        try:
            web_data = Path(web_data)

            #Make sure the path is to an actual file:
            if not web_data.is_file():
                wmsg = f"The provided web data path '{web_data}'"
                wmsg += " either doesn't exist or is not a file."
                wmsg += "\nNot sure what to do, so will skip this"
                wmsg += " particular web entry."
                print(wmsg)
                return
            #End if

        except TypeError:
            bad_input = False
            if _PANDAS_DF:
                if not isinstance(web_data, pd.DataFrame):
                    bad_input = True
                else:
                    data_frame = True
                #End if
            else:
                bad_input = True
            #End if
            if bad_input:
                wmsg = "WARNING: Inputs to 'add_website_data' can currently"
                wmsg += " only be paths to files or Pandas Dataframes, not"
                wmsg += f" type '{type(web_data).__name__}'"
                wmsg += "\nSkipping this website data entry..."
                print(wmsg)
                return
            #End if
        #End except

        #If multi-case and more than one case in ADF run, then
        #set to "multi-case".  Otherwise set to first case:
        if multi_case and not case_name:
            if self.num_cases > 1:
                case_name = "multi-case"
            else:
                case_name = self.get_cam_info("cam_case_name")[0]
            #End if
        #End if

        #Create HTML file path variable,
        #which will be used in "create_website":
        if data_frame:
            #Use web data name instead of case name for tables:
            html_name = f"amwg_table_{web_name}.html"

            #If multi-case, then save under the "multi-case" directory:
            if self.num_cases > 1:
                multi_tbl_dir = self.__case_web_paths['multi-case']["table_pages_dir"]
                #Avoid collecting individual case comparison tables
                #Hacky - could probably use an update eventually - JR
                if web_name != "case_comparison":
                    html_file.append(multi_tbl_dir/ html_name)
                html_file.append(self.__case_web_paths[case_name]["table_pages_dir"] / html_name)
            else:
                html_file = self.__case_web_paths[case_name]["table_pages_dir"] / html_name
            #End if
            asset_path = None
        else:
            html_name = f'plot_page_{web_data.stem}.html'
            html_file = self.__case_web_paths[case_name]["img_pages_dir"] / html_name
            asset_path = self.__case_web_paths[case_name]['assets_dir'] / web_data.name
        #End if

        #Initialize web data object:
        web_data = _WebData(web_data, web_name, case_name, plot_ext,
                            category = category,
                            season = season,
                            plot_type = plot_type,
                            data_frame = data_frame,
                            html_file = html_file,
                            asset_path = asset_path,
                            multi_case = multi_case)

        #Add web data object to list:
        self.__website_data.append(web_data)

        #Add plot type to plot order list:
        if (multi_case or data_frame) and self.num_cases > 1: #Actual multi-case
            if plot_type not in self.__plot_type_multi:
                self.__plot_type_multi.append(plot_type)
            #End if
            if plot_type not in self.__plot_type_order:
                self.__plot_type_order.append(plot_type)
            #End if
        else: #single case plot/ADF run
            if plot_type not in self.__plot_type_order:
                self.__plot_type_order.append(plot_type)
            #End if
        #End if

    #########

    def create_website(self):

        """
        Generate webpages to display diagnostic results.
        """

        #import needed standard modules:
        import shutil
        from collections import OrderedDict

        #Import "special" modules:
        try:
            import jinja2
        except ImportError:
            emsg = "Jinja2 module does not exist in python path, but is needed for website."
            emsg += "\nPlease install module, e.g. 'pip install Jinja2'"
            self.end_diag_fail(emsg)
        #End except

        #Notify user that script has started:
        print("\n  Generating Diagnostics webpages...")

        #If there is more than one non-baseline case, then create new website directory:
        if self.num_cases > 1:
            #Grab all multi-case diagnostic directories
            main_site_path = self.main_site_paths["main_site_path"]
            main_site_assets_path = self.main_site_paths["main_site_assets_path"]
            main_site_img_path = self.main_site_paths["main_site_img_path"]

            case_sites = OrderedDict()
            multi_layout = True
        else:
            main_site_path = "" #Set main_site_path to blank value
            multi_layout = False
        #End if

        #Extract needed variables from yaml file:
        case_names = self.get_cam_info('cam_case_name', required=True)

        #Grab case climo years
        syear_cases = self.climo_yrs["syears"]
        eyear_cases = self.climo_yrs["eyears"]

        #Grab baseline years (which may be empty strings if using Obs):
        syear_baseline = self.climo_yrs["syear_baseline"]
        eyear_baseline = self.climo_yrs["eyear_baseline"]

        #Set name of comparison data, which depends on "compare_obs":
        if self.compare_obs:
            data_name = "Obs"
            baseline_yrs = ""
            syear_baseline = ""
            eyear_baseline = ""
        else:
            data_name = self.get_baseline_info('cam_case_name', required=True)

            baseline_yrs=f"{syear_baseline} - {eyear_baseline}"
        #End if

        #Set climo years format for html file headers
        case_yrs=f"{syear_cases[0]} - {eyear_cases[0]}"

        #Extract variable defaults dictionary (for categories):
        var_defaults_dict = self.variable_defaults

        #Extract requested multi-case multi-plots
        multi_case_plots = self.read_config_var('multi_case_plots')

        if multi_case_plots:
            #Grab all variables for each multi-case plot type
            mvars = []
            #ext are plot type extentions (keys for multi-case plots)
            #var_list should be a list of all vars for each plot map extentions
            #var is iterative for all plot map extensions
            for multi_var_list in [multi_case_plots[ext] for ext in multi_case_plots]:
                for var in multi_var_list:
                    if (self.compare_obs and var in self.var_obs_dict) or (not self.compare_obs):
                        mvars.append(var)

        #Create multi-case site:
        #Make a dictionary for plot type extensions for given plot type
        #This can probably be populated in the for-loops during html creation...
        #Or it should be declared somewhere higher up, like adf_info or something
        multi_case_dict = {"global_latlon_map":"LatLon",
                               "zonal_mean":"Zonal",
                               "meridional":"Meridional",
                               "global_latlon_vect_map":"LatLon_Vector"}

        #Dictionary for multi-case website plot types
        multi_plots = {"Tables": "html_table/mean_tables.html",
                       "Special":"html_img/multi_case_mean_diag_Special.html"}

        #Set plot type html dictionary (for Jinja templating):
        plot_type_html = OrderedDict()
        for plot_type in self.__plot_type_order:
            if plot_type == 'Tables':
                plot_type_html[plot_type] = os.path.join("html_table", "mean_tables.html")
            else:
                plot_type_html[plot_type] = os.path.join("html_img", f"mean_diag_{plot_type}.html")
            #End if
        #End for

        #Do the same for multi-case:
        if self.num_cases > 1:
            multi_plot_type_html = OrderedDict()
            for plot_type in self.__plot_type_multi:
                if plot_type == 'Tables':
                    multi_plot_type_html[plot_type] = os.path.join("html_table",
                                                                   "mean_tables.html")
                else:
                    multi_plot_type_html[plot_type] = os.path.join("html_img",
                                                        f"multi_case_mean_diag_{plot_type}.html")
                #End if
            #End for

        #Set main title for website:
        main_title = "CAM Diagnostics"

        #Determine local directory:
        adf_lib_dir = Path(__file__).parent

        #Set path to Jinja2 template files:
        jinja_template_dir = Path(adf_lib_dir, 'website_templates')

        #Create the jinja Environment object:
        jinenv = jinja2.Environment(loader=jinja2.FileSystemLoader(jinja_template_dir))

        # this is going to hold the data for building the mean
        # plots provisional structure:
        # key = variable_name
        # values -> dict w/ keys being "TYPE" of plots
        # w/ values being dict w/ keys being TEMPORAL sampling,
        # values being the URL
        #Note: It might be better if the "mean_html_info"
        #dictionary was created in the "add_website_data",
        #so that we only had to do the web_data loop once,
        #but for now this will do. -JN
        mean_html_info = OrderedDict()
        #Use this for multi-case diagnostic plots only
        multi_mean_html_info = OrderedDict()
        #Use this for multi-case with multi-plots
        multi_plot_html_info = OrderedDict()

        #Create another dictionary needed for HTML pages that render tables:
        table_html_info = OrderedDict()
        #Use this for multi-case diagnostic tables
        multi_table_html_info = OrderedDict()

        #If this is a multi-case instance, then copy website to "main" directory:
        if main_site_path:
            self.__case_web_paths["multi-case"]['img_pages_dir'].mkdir(exist_ok=True)

            #Create CSS templates file path:
            main_templates_path = main_site_path / "templates"

            #Also add path to case_sites dictionary:
            #loop over cases:
            for idx, case_name in enumerate(case_names):
                #Check if case name is present in plot
                if case_name in self.__case_web_paths:
                    #Add path to case_sites dictionary:
                    case_dir_ext = f"{case_name}_{syear_cases[idx]}_{eyear_cases[idx]}"
                    if self.compare_obs:
                        base_dir_ext = f"{data_name}"
                    else:
                        base_dir_ext = f"{data_name}_{syear_baseline}_{eyear_baseline}"
                    case_sites[case_name] = [os.path.join(os.curdir,
                                             f"{case_dir_ext}_vs_{base_dir_ext}",
                                             "index.html"),syear_cases[idx],eyear_cases[idx]]

        else:
            #make empty list for non multi-case web generation
            case_sites = []

        #Loop over all web data objects:
        for web_data in self.__website_data:

            #Create the directory where the website will be built:
            self.__case_web_paths[web_data.case]['website_dir'].mkdir(exist_ok=True)

            #Create a directory where CSS files will be stored:
            css_files_dir = self.__case_web_paths[web_data.case]['css_files_dir']
            css_files_dir.mkdir(exist_ok=True)

            #Copy CSS files over to output directory:
            for css_file in jinja_template_dir.glob('*.css'):
                shutil.copyfile(css_file, css_files_dir / css_file.name)
            #End for

            #Copy GIF files over to output directory as well:
            for gif_file in jinja_template_dir.glob('*.gif'):
                shutil.copyfile(gif_file, css_files_dir / gif_file.name)
            #End for

            #Check first for AMWG tables data frame
            if web_data.data_frame:

                #Create a directory that will hold table html files, if a table is present:
                if self.num_cases > 1:
                    self.__case_web_paths['multi-case']['table_pages_dir'].mkdir(exist_ok=True)
                #End if

                self.__case_web_paths[web_data.case]['table_pages_dir'].mkdir(exist_ok=True)

                #Add table HTML file to dictionary:
                #Note:  Need to use data name instead of case name for tables.
                if len(case_names) > 1:
                    if web_data.name != "case_comparison":
                        table_html_info[web_data.name] = web_data.html_file[0].name

                    multi_table_html_info[web_data.name] = web_data.html_file[0].name
                else:
                    table_html_info[web_data.name] = web_data.html_file.name

            #Now check all plot types
            if not web_data.data_frame:
                #Determine season value:
                if web_data.season:
                    season = web_data.season
                else:
                    season = "plot" #Just have the link be labeled "plot".
                #End if

                #Extract plot_type:
                ptype = web_data.plot_type

                #Extract web data name (usually the variable name):
                var = web_data.name

                #Check if category has been provided for this web data:
                if web_data.category:
                    #If so, then just use directly:
                    category = web_data.category
                else:
                    #Check if variable in defaults dictionary:
                    if web_data.name in var_defaults_dict:
                        #If so, then extract category from dictionary:
                        category = var_defaults_dict[web_data.name].get("category",
                                                                        "No category yet")
                    else:
                        category = 'No category yet'
                    #End if
                #End if

                #Check to see if there are multiple-cases
                if main_site_path:
                    #Check to see if the user has multi-plots enabled
                    if multi_case_plots:
                        #Loop over each variable in multi-case plot variables
                        #Check if plot ext is in requested multi-case plot types
                        if (web_data.plot_ext in multi_case_plots.keys()) and (var in mvars):

                            #Initialize Ordered Dictionary for multi case plot type:
                            if ptype not in multi_plot_html_info:
                                multi_plot_html_info[ptype] = OrderedDict()
                            #End if
                            #Initialize Ordered Dictionary for category:
                            if category not in multi_plot_html_info[ptype]:
                                multi_plot_html_info[ptype][category] = OrderedDict()
                            #End if
                            if var not in multi_plot_html_info[ptype][category]:
                                multi_plot_html_info[ptype][category][var] = OrderedDict()
                            #End if
                            p = f"plot_page_multi_case_{var}_{season}_{ptype}_Mean.html"
                            if season not in multi_plot_html_info[ptype][category][var]:
                                multi_plot_html_info[ptype][category][var][season] = p
                            #End if
                        #End if (variable in multi-case plot variables)
                    #End if multi-case multi-plots

                    #Need to isolate multi-case regular plots from the multi-case multi-plots
                    #QUESTION: Is there a better way?
                    if "multi_plot" not in str(web_data.html_file.name):
                        if ptype not in multi_mean_html_info:
                            multi_mean_html_info[ptype] = OrderedDict()
                        #End if
                        if category not in multi_mean_html_info[ptype]:
                            multi_mean_html_info[ptype][category] = OrderedDict()
                        #End if
                        if var not in multi_mean_html_info[ptype][category]:
                            multi_mean_html_info[ptype][category][var] = OrderedDict()
                        #End if
                        p = f"plot_page_multi_case_{var}_{season}_{ptype}_Mean.html"
                        if season not in multi_mean_html_info[ptype][category][var]:
                            multi_mean_html_info[ptype][category][var][season] = p
                        #End if
                    #End if (not multi-case multi-plots)
                #End if (multi-case scenario)

                #Individual cases
                #This will be used if multi-case diagnostics as well
                #Create a directory that will hold just the html files for individual images:
                self.__case_web_paths[web_data.case]['img_pages_dir'].mkdir(exist_ok=True)

                #Create a directory that will hold copies of the actual images:
                self.__case_web_paths[web_data.case]['assets_dir'].mkdir(exist_ok=True)

                #Move file to assets directory:
                try:
                    shutil.copy(web_data.data, web_data.asset_path)
                except:
                    pass

                #Initialize Ordered Dictionary for season:
                #Need to ignore plots that may be generated for multi-case diags
                #NOTE: There is probably a better way to do this - JR
                if "multi_plot" not in str(web_data.html_file.name):
                    #Initialize Ordered Dictionary for plot type:
                    if ptype not in mean_html_info:
                        mean_html_info[ptype] = OrderedDict()
                    #End if
                    if category not in mean_html_info[ptype]:
                        mean_html_info[ptype][category] = OrderedDict()
                    #End if
                    #Initialize Ordered Dictionary for variable:
                    if var not in mean_html_info[ptype][category]:
                        mean_html_info[ptype][category][var] = OrderedDict()
                    #End if
                    mean_html_info[ptype][category][var][season] = web_data.html_file.name
                #End if (check for multi_plot)
            #End if (data-frame check)
        #End for (web_data list loop)

        #Loop over all web data objects again:
        #NOTE: This will be for non multi-case diagnostics
        for idx,web_data in enumerate(self.__website_data):
            if web_data.data_frame:
                #Create output HTML file path:
                if self.num_cases > 1:
                    table_pages_dir = self.__case_web_paths['multi-case']['table_pages_dir']
                    table_pages_dir_indv = self.__case_web_paths[web_data.case]['table_pages_dir']

                else:
                    table_pages_dir = self.__case_web_paths[web_data.case]['table_pages_dir']
                #End if

                #Check if plot image already handles multiple cases,
                #and if so change the case name:
                if web_data.multi_case:
                    case1 = "Listed in tables"
                else:
                    case1 = web_data.case
                #End if

                #Write table dataframe HTML as a string:
                #Note:  One could generate an image file here instead of raw HTML code,
                #which might be beneficial for colored tables and other more advance
                #formatting features.
                table_html = web_data.data.to_html(index=False, border=1, justify='center',
                                                   float_format='{:6g}'.format)

                #Construct amwg_table.html
                rend_kwarg_dict = {"title": main_title,
                                  "case_name": case1,
                                  "case_yrs": case_yrs,
                                  "base_name": data_name,
                                  "baseline_yrs": baseline_yrs,
                                  "amwg_tables": table_html_info,
                                  "table_name": web_data.name,
                                  "table_html": table_html,
                                  "multi_head": False,
                                  "multi": multi_layout,
                                  "case_sites": case_sites}

                table_tmpl = jinenv.get_template('template_table.html')

                if main_site_path:
                    #Avoid single case comparison getting called here
                    #There might be a better way, but for now it works - JR
                    if web_data.name != "case_comparison":
                        rend_kwarg_dict["plot_types"] = multi_plot_type_html
                        rend_kwarg_dict["multi_head"] = "Table"

                        table_rndr = table_tmpl.render(rend_kwarg_dict)

                        #Write mean diagnostic tables HTML file:
                        html_file = web_data.html_file[0]
                        with open(html_file, 'w', encoding='utf-8') as ofil:
                            ofil.write(table_rndr)

                else:
                    rend_kwarg_dict["plot_types"] = plot_type_html
                    if web_data.case == data_name:
                        rend_kwarg_dict["case_name"] = case_names[0]

                    table_rndr = table_tmpl.render(rend_kwarg_dict)

                    #Write mean diagnostic tables HTML file:
                    html_file = web_data.html_file
                    with open(html_file, 'w', encoding='utf-8') as ofil:
                        ofil.write(table_rndr)


                #Check if the mean plot type page exists for this case (or for multi-case):
                mean_table_file = table_pages_dir / "mean_tables.html"
                if not mean_table_file.exists():

                    #Construct mean_table.html
                    mean_table_tmpl = jinenv.get_template('template_mean_tables.html')

                    #Reuse the rend_kwarg_dict, but ignore certain keys
                    #since all others are the same
                    new_dict = {k: rend_kwarg_dict[k] for k in rend_kwarg_dict.keys() - {'table_name', 'table_html'}}    

                    if main_site_path:
                        plot_types = multi_plot_type_html
                        new_dict["multi_head"] = "Table"
                    else:
                        plot_types = plot_type_html

                    mean_table_rndr = mean_table_tmpl.render(new_dict)

                    #Write mean diagnostic tables HTML file:
                    with open(mean_table_file, 'w', encoding='utf-8') as ofil:
                        ofil.write(mean_table_rndr)
                    #End with
                #End if
            #End if (tables)

            else: #Plot image
                plot_types = plot_type_html

                #Ensure that these will ignore any multi-case pages if they exist
                if "main_website" not in str(web_data.html_file):
                    #Create output HTML file path:
                    img_pages_dir = self.__case_web_paths[web_data.case]['img_pages_dir']

                    img_data = [os.path.relpath(web_data.asset_path, start=img_pages_dir),
                            web_data.asset_path.stem]

                    rend_kwarg_dict = {"title": main_title,
                                       "var_title": web_data.name,
                                       "season_title": web_data.season,
                                       "case_name": web_data.case,
                                       "case_yrs": case_yrs,
                                       "base_name": data_name,
                                       "baseline_yrs": baseline_yrs,
                                       "plottype_title": web_data.plot_type,
                                       "imgs": img_data,
                                       "mydata": mean_html_info[web_data.plot_type],
                                       "plot_types": plot_types,
                                       "multi": multi_layout}

                    tmpl = jinenv.get_template('template.html')  #Set template

                    rndr = tmpl.render(rend_kwarg_dict) #The template rendered

                    #Write HTML file:
                    with open(web_data.html_file, 'w', encoding='utf-8') as ofil:
                        ofil.write(rndr)
                    #End with

                    #Check if the mean plot type page exists for this case:
                    mean_ptype_file = img_pages_dir / f"mean_diag_{web_data.plot_type}.html"
                    if not mean_ptype_file.exists():

                        #Construct individual plot type mean_diag html files, if they don't
                        #already exist:
                        mean_tmpl = jinenv.get_template('template_mean_diag.html')

                        #Remove keys from main dictionary for this html page
                        templ_rend_kwarg_dict = {k: rend_kwarg_dict[k] for k in rend_kwarg_dict.keys() - {'imgs', 'var_title', 'season_title'}}

                        mean_rndr = mean_tmpl.render(templ_rend_kwarg_dict)

                        #Write mean diagnostic plots HTML file:
                        with open(mean_ptype_file,'w', encoding='utf-8') as ofil:
                            ofil.write(mean_rndr)
                        #End with
                    #End if (mean_ptype exists)

                    #Check if the mean plot type and var page exists for this case:
                    plot_page = f"plot_page_{web_data.name}_{web_data.plot_type}.html"
                    mean_ptype_plot_page = img_pages_dir / plot_page
                    if not mean_ptype_plot_page.exists():

                        #Construct individual plot type mean_diag html files, if they don't
                        #already exist:
                        plot_page_tmpl = jinenv.get_template('template_var.html')

                        #Remove key from main dictionary for this html page
                        templ_var_rend_kwarg_dict = {k: rend_kwarg_dict[k] for k in rend_kwarg_dict.keys() - {'imgs'}}

                        plot_page_rndr = plot_page_tmpl.render(templ_var_rend_kwarg_dict)

                        #Write mean diagnostic plots HTML file:
                        with open(mean_ptype_plot_page,'w', encoding='utf-8') as ofil:
                            ofil.write(plot_page_rndr)
                        #End with
                    #End if (mean_ptype_plot_page)
                #End if (check for multi-case diags)
            #Also check if index page exists for this case:
            index_html_file = \
                self.__case_web_paths[web_data.case]['website_dir'] / "index.html"

            #Re-et plot types list:
            if web_data.case == 'multi-case':
                plot_types = multi_plot_type_html
            else:
                plot_types = plot_type_html
            plot_types = plot_type_html
            #End if

            #Construct index.html
            index_title = "AMP Diagnostics Prototype"
            index_tmpl = jinenv.get_template('template_index.html')

            index_rndr = index_tmpl.render(title=index_title,
                                            case_name=web_data.case,
                                            base_name=data_name,
                                            case_yrs=case_yrs,
                                            baseline_yrs=baseline_yrs,
                                            plot_types=plot_types,
                                            multi=multi_layout)

            #Write Mean diagnostics index HTML file:
            with open(index_html_file, 'w', encoding='utf-8') as ofil:
                ofil.write(index_rndr)
            #End with

            #End if (plot images)
        #End for (web data loop)

        # - - - - - - - - - - - - - - - - - - - - - -
        # --- End single-case diagnostics ---
        # - - - - - - - - - - - - - - - - - - - - - -



        # - - - - - - - - - - - - - - - - - - - - - -
        # --- Checking for multi-case diagnostics ---
        # - - - - - - - - - - - - - - - - - - - - - -

        if main_site_path:
            #Loop over all web data objects AGAIN:
            for web_data in self.__website_data:

                #Create CSS templates file path:
                main_templates_path = main_site_path / "templates"

                #loop over all cases and make website directories:
                for idx, case_name in enumerate(case_names):
                    #Check if case name is present in plot
                    if case_name in self.__case_web_paths:
                        #Extract website directory:
                        website_dir = self.__case_web_paths[case_name]['website_dir']
                        if not website_dir.is_dir():
                            website_dir.mkdir(parents=True)
                            #Copy website directory to "main site" directory:
                            shutil.copytree(website_dir, main_site_path / case_name)


                #Starting multi-case tables if requested
                # - - - - - - - - - - - - - - - - - - -
                if web_data.data_frame:

                    #Create all individual tables for the individual websites

                    #Grab single case table path
                    table_pages_dir_indv = self.__case_web_paths[web_data.case]['table_pages_dir']

                    #Check if the mean plot type page exists for this case (or for multi-case):
                    mean_table_file = table_pages_dir_indv / "mean_tables.html"
                    table_keys = [web_data.case,data_name,"case_comparison"]

                    table_dict = {}
                    for key in table_keys:
                        if self.compare_obs:
                            if (key != "Obs") and (key != "case_comparison"):
                                table_dict[key] = multi_table_html_info[key]
                        else:
                            table_dict[key] = multi_table_html_info[key]
                    #End for

                    #Construct amwg_table.html
                    rend_kwarg_dict = {"title": main_title, "case_name": web_data.case,
                                        "case_yrs": case_yrs,
                                        "base_name": data_name,
                                        "baseline_yrs": baseline_yrs,
                                        "amwg_tables": table_dict,
                                        "table_name": web_data.name,
                                        "plot_types": plot_type_html,
                                        "multi_head": True,
                                        "multi": False,
                                        "case_sites": case_sites}

                    if not mean_table_file.exists():
                        #Construct mean_table.html
                        mean_table_tmpl = jinenv.get_template('template_mean_tables.html')

                        #Remove key from main dictionary for this html page
                        tmpl_rend_kwarg_dict = {k: rend_kwarg_dict[k] for k in rend_kwarg_dict.keys() - {'table_name'}}

                        mean_table_rndr = mean_table_tmpl.render(tmpl_rend_kwarg_dict)

                        #Write mean diagnostic tables HTML file:
                        with open(mean_table_file, 'w', encoding='utf-8') as ofil:
                            ofil.write(mean_table_rndr)
                        #End with
                    #End if

                    #Loop through all test cases (exclude baseline)
                    if web_data.case != data_name:
                        table_html = web_data.data.to_html(index=False, border=1, justify='center',
                                                            float_format='{:6g}'.format)

                        indv_html = table_pages_dir_indv / f"amwg_table_{web_data.name}.html"

                        if not indv_html.exists():
                            table_tmpl = jinenv.get_template('template_table.html')

                            tmpl_rend_kwarg_dict = rend_kwarg_dict
                            rend_kwarg_dict["table_html"] = table_html

                            table_rndr = table_tmpl.render(rend_kwarg_dict)

                            #Write mean diagnostic tables HTML file:
                            with open(indv_html, 'w', encoding='utf-8') as ofil:
                                ofil.write(table_rndr)
                        #end if (indv_html)

                    #Baseline case added to all test case directories
                    # - this block should only run once when web_data is the baseline case
                    else:
                        table_html = web_data.data.to_html(index=False, border=1, justify='center',
                                                            float_format='{:6g}'.format)

                        rend_kwarg_dict["table_html"] = table_html

                        for case_name in case_names:
                            table_pages_dir_sp = self.__case_web_paths[case_name]['table_pages_dir']
                            table_key = [case_name,data_name,"case_comparison"]

                            base_table_dict = {key: multi_table_html_info[key] for key in table_key}

                            rend_kwarg_dict["case_name"] = case_name
                            rend_kwarg_dict["amwg_tables"] = base_table_dict

                            sp_html = table_pages_dir_sp / f"amwg_table_{data_name}.html"
                            if not sp_html.exists():

                                table_tmpl = jinenv.get_template('template_table.html')

                                table_rndr = table_tmpl.render(rend_kwarg_dict)

                                with open(sp_html, 'w', encoding='utf-8') as ofil:
                                    ofil.write(table_rndr)
                            #End if (sp_html)
                        #End for (case_names)
                    #End if (baseline_name)


                #Starting multi-case plots if requested
                # - - - - - - - - - - - - - - - - - - -
                if not web_data.data_frame:

                    #Extract plot details
                    season = web_data.season
                    ptype = web_data.plot_type
                    var = web_data.name
                    ext = web_data.plot_ext

                    #Check if category has been provided for this web data:
                    if web_data.category:
                        #If so, then just use directly:
                        category = web_data.category
                    else:
                        #Check if variable in defaults dictionary:
                        if web_data.name in var_defaults_dict:
                            #If so, then extract category from dictionary:
                            category = var_defaults_dict[web_data.name].get("category",
                                                                            "No category yet")
                        else:
                            category = 'No category yet'
                        #End if
                    #End if

                    #Check for multi-case multi-plots
                    if multi_case_plots:
                        #This currently runs web_data.case for every case, but in reality
                        #it really only needs to run once since the plots are
                        #already made with all cases.
                        #So just grab the first test case:
                        case1 = self.get_cam_info('cam_case_name', required=True)[0]
                        if str(web_data.case) == str(case1):
                            #Check if variable is in desired multi-case plot
                            #and if plot_type is in given multi-case plot set:
                            if (var in mvars) and (ext in multi_case_plots):
                                #Move file to assets directory:
                                if not web_data.data.is_file():
                                    shutil.copy(web_data.data, web_data.asset_path)

                                #Create output HTML file path:
                                img_pages_dir = self.__case_web_paths["multi-case"]['img_pages_dir']
                                multi_plot_page = f"{var}_{season}_{ptype}_multi_plot.png"
                                img_data = [os.path.relpath(main_site_assets_path / multi_plot_page,
                                                        start=main_site_img_path),
                                                        multi_plot_page]

                                rend_kwarg_dict = {"title": main_title,
                                                    "var_title": var,
                                                    "season_title": season,
                                                    "case_yrs": case_yrs,
                                                    "base_name": data_name,
                                                    "baseline_yrs": baseline_yrs,
                                                    "plottype_title": ptype,
                                                    "imgs": img_data,
                                                    "mydata": multi_plot_html_info[ptype],
                                                    "plot_types": multi_plot_type_html,
                                                    "multi": multi_layout,
                                                    "case_sites": case_sites} 

                                multimean = f"plot_page_multi_case_{var}_{season}_{ptype}_Mean.html"
                                if not (img_pages_dir / multimean).exists():

                                    tmpl = jinenv.get_template('template_multi_case.html')

                                    rndr = tmpl.render(rend_kwarg_dict)

                                    #Write HTML file:
                                    with open(img_pages_dir / multimean,
                                            'w', encoding='utf-8') as ofil:
                                        ofil.write(rndr)
                                #End if (multimean)

                                #Check if the mean plot type and var page exists for this case:
                                img_pages_dir = self.__case_web_paths["multi-case"]['img_pages_dir']
                                plot_page = f"plot_page_multi_case_{var}_{ptype}.html"
                                mean_ptype_plot_page = img_pages_dir / plot_page

                                if not mean_ptype_plot_page.exists():

                                    #Remove key from main dictionary for this html page
                                    templ_var_rend_kwarg_dict = {k: rend_kwarg_dict[k] for k in rend_kwarg_dict.keys() - {'imgs'}}

                                    #Construct individual plot type mean_diag
                                    #html files, if they don't already exist:
                                    page_tmpl = jinenv.get_template('template_multi_case_var.html')

                                    plot_page_rndr = page_tmpl.render(templ_var_rend_kwarg_dict)

                                    #Write mean diagnostic plots HTML file:
                                    with open(mean_ptype_plot_page,'w', encoding='utf-8') as ofil:
                                        ofil.write(plot_page_rndr)
                                    #End with
                                #End if (mean_ptype_plot_page exists)

                                multi_mean = f"multi_case_mean_diag_{ptype}.html"
                                mean_ptype_file = main_site_img_path / multi_mean
                                if not mean_ptype_file.exists():

                                    #Remove keys from main dictionary for this html page
                                    templ_rend_kwarg_dict = {k: rend_kwarg_dict[k] for k in rend_kwarg_dict.keys() - {'imgs', 'var_title', 'season_title'}}

                                    #Construct individual plot type mean_diag
                                    #html files, if they don't already exist:
                                    tmp = jinenv.get_template('template_multi_case_mean_diag.html')

                                    mean_rndr = tmp.render(templ_rend_kwarg_dict)

                                    #Write mean diagnostic plots HTML file:
                                    with open(mean_ptype_file,'w', encoding='utf-8') as ofil:
                                        ofil.write(mean_rndr)
                                    #End with
                                #End if (mean_ptype exists)

                    #Loop over any non multi-case multi-plot scenarios
                    #ie multi-case Taylor Diagrams and multi-case QBO
                    if ext not in multi_case_dict:
                        #Move file to assets directory:
                        if not web_data.data.is_file():
                            shutil.copy(web_data.data, web_data.asset_path)

                        #Create output HTML file path:
                        img_pages_dir = self.__case_web_paths["multi-case"]['img_pages_dir']
                        multi_plot_page = f"{var}_{season}_{ptype}_multi_plot.png"
                        img_data = [os.path.relpath(main_site_assets_path / multi_plot_page,
                                                            start=main_site_img_path),
                                                            multi_plot_page]

                        rend_kwarg_dict = {"title": main_title,
                                            "var_title": var,
                                            "season_title": season,
                                            "case_yrs": case_yrs,
                                            "base_name": data_name,
                                            "baseline_yrs": baseline_yrs,
                                            "plottype_title": ptype,
                                            "imgs": img_data,
                                            "mydata": multi_mean_html_info[ptype],
                                            "plot_types": multi_plot_type_html,
                                            "multi": multi_layout,
                                            "case_sites": case_sites}                                    

                        multimean = f"plot_page_multi_case_{var}_{season}_{ptype}_Mean.html"
                        if not (img_pages_dir / multimean).exists():
                            tmpl = jinenv.get_template('template_multi_case.html')

                            rndr = tmpl.render(rend_kwarg_dict)

                            #Write HTML file:
                            with open(img_pages_dir / multimean,
                                                'w', encoding='utf-8') as ofil:
                                ofil.write(rndr)
                        #End if (multimean)

                        #Check if the mean plot type and var page exists for this case:
                        img_pages_dir = self.__case_web_paths["multi-case"]['img_pages_dir']
                        plot_page = f"plot_page_multi_case_{var}_{ptype}.html"
                        mean_ptype_plot_page = img_pages_dir / plot_page

                        if not mean_ptype_plot_page.exists():

                            #Remove key from main dictionary for this html page
                            templ_var_rend_kwarg_dict = {k: rend_kwarg_dict[k] for k in rend_kwarg_dict.keys() - {'imgs'}}

                            #Construct individual plot type mean_diag
                            #html files, if they don't already exist:
                            page_tmpl = jinenv.get_template('template_multi_case_var.html')

                            plot_page_rndr = page_tmpl.render(templ_var_rend_kwarg_dict)

                            #Write mean diagnostic plots HTML file:
                            with open(mean_ptype_plot_page,'w', encoding='utf-8') as ofil:
                                ofil.write(plot_page_rndr)
                            #End with
                        #End if (mean_ptype_plot_page exists)

                        multi_mean = f"multi_case_mean_diag_{ptype}.html"
                        mean_ptype_file = main_site_img_path / multi_mean
                        if not mean_ptype_file.exists():

                            #Remove keys from main dictionary for this html page
                            templ_rend_kwarg_dict = {k: rend_kwarg_dict[k] for k in rend_kwarg_dict.keys() - {'imgs', 'var_title', 'season_title'}}

                            #Construct individual plot type mean_diag
                            #html files, if they don't already exist:
                            tmp = jinenv.get_template('template_multi_case_mean_diag.html')

                            mean_rndr = tmp.render(templ_rend_kwarg_dict)

                            #Write mean diagnostic plots HTML file:
                            with open(mean_ptype_file,'w', encoding='utf-8') as ofil:
                                ofil.write(mean_rndr)
                            #End with
                        #End if (mean_ptype exists)
                    #End if (ext not in multi_case_dict)
                #End if (web_data.data_frame)
            #End for (web_data)

            #Also make sure CSS template files have been copied over:
            if not main_templates_path.is_dir():
                #If not, just grab the files from the first test case directory
                css_files_dir = self.__case_web_paths[case_names[-1]]['css_files_dir']
                shutil.copytree(css_files_dir, main_templates_path)
            #End if

            if multi_case_plots:
                for key in multi_case_plots:
                    #Update the dictionary to add any plot types specified in the yaml file
                    mcase_plot = f"html_img/multi_case_mean_diag_{multi_case_dict[key]}.html"
                    multi_plots[multi_case_dict[key]] = mcase_plot
                #End for
            #End if

            main_title = "ADF Diagnostics"
            main_tmpl = jinenv.get_template('template_multi_case_index.html')
            main_rndr = main_tmpl.render(title=main_title,
                                         case_sites=case_sites,
                                         base_name=data_name,
                                         baseline_yrs=baseline_yrs,
                                         multi_plots=multi_plots)

            #Write multi-case main HTML file:
            outputfile = main_site_path / "index.html"
            with open(outputfile, 'w', encoding='utf-8') as ofil:
                ofil.write(main_rndr)
            #End with
        #End if (multi case)

        #Notify user that script has finishedd:
        print("  ...Webpages have been generated successfully.")

#++++++++++++++++++++
#End Class definition
#++++++++++++++++++++