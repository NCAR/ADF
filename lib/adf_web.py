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
import markdown

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
                 category = None,
                 season = None,
                 non_season = False,
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
        self.non_season = non_season
        self.plot_type  = plot_type
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

            #Add links to external packages (if applicable)
            self.external_package_links = {}

            #MDTF puts directory under case[0]
            if self.get_mdtf_info('mdtf_run'):
                syear = self.climo_yrs["syears"]
                eyear = self.climo_yrs["eyears"]
                mdtf_path = f"../mdtf/MDTF_{case_name}"
                mdtf_path += f"_{syear[0]}_{eyear[0]}"
                self.external_package_links['MDTF'] = mdtf_path
            #End if
            
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
                         category = None,
                         season = None,
                         non_season = False,
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

        non_season -> Are the plots NOT divided up by seaons, ANN, DJF, MAM, JJA, or SON?
                      - QBO is displayed as QBOts and QBOamp in the season argument above

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
                html_file = self.__case_web_paths['multi-case']["table_pages_dir"] / html_name
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
        web_data = _WebData(web_data, web_name, case_name,
                            category = category,
                            season = season,
                            non_season = non_season,
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

        #Make jinja functions that mimics python functions.
        #  - This will allow for the use of 'list' in the html rendering.
        def jinja_list(seas_list):
            return list(seas_list)
        #   - This will allow for the use of 'enumerate' in the html rendering.
        def jinja_enumerate(arg):
            return enumerate(arg)

        #Notify user that script has started:
        print("\n  Generating Diagnostics webpages...")

        #If there is more than one non-baseline case, then create new website directory:
        if self.num_cases > 1:
            main_site_path = Path(self.get_basic_info('cam_diag_plot_loc', required=True))
            main_site_path = main_site_path / "main_website"
            main_site_path.mkdir(exist_ok=True)
            case_sites = OrderedDict()
        else:
            main_site_path = "" #Set main_site_path to blank value
        #End if

        #Access variable defaults yaml file
        res = self.variable_defaults

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

        else:
            data_name = self.get_baseline_info('cam_case_name', required=True)

            baseline_yrs=f"{syear_baseline} - {eyear_baseline}"
        #End if

        #Set climo years format for html file headers
        case_yrs=f"{syear_cases[0]} - {eyear_cases[0]}"

        #Extract variable defaults dictionary (for categories):
        var_defaults_dict = self.variable_defaults

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
                                                                   f"mean_diag_{plot_type}.html")
                #End if
            #End for
        else:
            #Set to match standard plot type dict:
            multi_plot_type_html = plot_type_html
        #End if

        #Set main title for website:
        main_title = "CAM Diagnostics"

        #List of seasons
        seasons = ["ANN","DJF","MAM","JJA","SON"]

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

        non_seasons = OrderedDict()

        #Create another dictionary needed for HTML pages that render tables:
        table_html_info = OrderedDict()

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
                else:
                    self.__case_web_paths[web_data.case]['table_pages_dir'].mkdir(exist_ok=True)
                #End if

                #Add table HTML file to dictionary:
                #Note:  Need to use data name instead of case name for tables.
                table_html_info[web_data.name] = web_data.html_file.name

           #Now check all plot types
            if not web_data.data_frame:
                #Determine season value:
                if web_data.season:
                    season = web_data.season
                else:
                    season = "plot" #Just have the link be labeled "plot".
                #End if

                #Extract web data name (usually the variable name):
                var = web_data.name

                #Extract whether plot has traditional seasons or not
                non_season = web_data.non_season

                #Create a directory that will hold just the html files for individual images:
                self.__case_web_paths[web_data.case]['img_pages_dir'].mkdir(exist_ok=True)

                #Create a directory that will hold copies of the actual images:
                self.__case_web_paths[web_data.case]['assets_dir'].mkdir(exist_ok=True)

                #Move file to assets directory:
                shutil.copy(web_data.data, web_data.asset_path)

                #Extract plot_type:
                ptype = web_data.plot_type

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

                #Initialize Ordered Dictionary for season:
                mean_html_info[ptype][category][var][season] = web_data.html_file.name


                #Initialize Ordered Dictionary for non season kwarg:
                if ptype not in non_seasons:
                    non_seasons[ptype] = OrderedDict()
                #End if
                if category not in non_seasons[ptype]:
                    non_seasons[ptype][category] = OrderedDict()
                #End if
                if var not in non_seasons[ptype][category]:
                    non_seasons[ptype][category][var] = non_season
                #End if

            #End if (data-frame check)
        #End for (web_data list loop)

        #Loop over all web data objects again:
        for web_data in self.__website_data:

            if web_data.data_frame:

                #Create output HTML file path:
                if self.num_cases > 1:
                    table_pages_dir = self.__case_web_paths['multi-case']['table_pages_dir']
                    plot_types = multi_plot_type_html
                else:
                    table_pages_dir = self.__case_web_paths[web_data.case]['table_pages_dir']
                    plot_types = plot_type_html
                #End if

                #Check if plot image already handles multiple cases,
                #and if so change the case name:
                if web_data.multi_case:
                    case1 = "Listed in tables"
                else:
                    case1 = case_names[0]
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
                                  "multi_head": False}
                rend_kwarg_dict["plot_types"] = multi_plot_type_html

                if web_data.name == case1:
                    rend_kwarg_dict["disp_table_name"] = case1
                    rend_kwarg_dict["disp_table_html"] = table_html
                
                if web_data.name == "Case Comparison":
                    rend_kwarg_dict["disp_table_name"] = "Case Comparison"
                    rend_kwarg_dict["disp_table_html"] = table_html

                table_tmpl = jinenv.get_template('template_table.html')
                table_rndr = table_tmpl.render(rend_kwarg_dict)

                #Write mean diagnostic tables HTML file:
                with open(web_data.html_file, 'w', encoding='utf-8') as ofil:
                    ofil.write(table_rndr)
                #End with

                #Check if the mean plot type page exists for this case (or for multi-case):
                mean_table_file = table_pages_dir / "mean_tables.html"

                #Construct mean_table.html
                mean_table_tmpl = jinenv.get_template('template_mean_tables.html')
                #Reuse the rend_kwarg_dict
                mean_table_rndr = mean_table_tmpl.render(rend_kwarg_dict)
                #Write mean diagnostic tables HTML file:
                with open(mean_table_file, 'w', encoding='utf-8') as ofil:
                    ofil.write(mean_table_rndr)
                #End with

            #End if (tables)

            else: #Plot image
                plot_types = plot_type_html

                #Create output HTML file path:
                img_pages_dir = self.__case_web_paths[web_data.case]['img_pages_dir']
                img_data = [os.path.relpath(web_data.asset_path, start=img_pages_dir),
                            web_data.asset_path.stem]
                #Check if plot image already handles multiple cases:
                if web_data.multi_case:
                    case1 = "Listed in plots."
                    plot_types = multi_plot_type_html
                else:
                    case1 = web_data.case
                    plot_types = plot_type_html
                #End if

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
                                   "seasons": seasons,
                                   "non_seasons": non_seasons[web_data.plot_type]}

                tmpl = jinenv.get_template('template.html')  #Set template
                rndr = tmpl.render(rend_kwarg_dict) #The template rendered

                #Write HTML file:
                with open(web_data.html_file, 'w', encoding='utf-8') as ofil:
                    ofil.write(rndr)
                #End with

                #Mean plot type html file name
                mean_ptype_file = img_pages_dir / f"mean_diag_{web_data.plot_type}.html"

                #Construct individual plot type mean_diag html files
                mean_tmpl = jinenv.get_template('template_mean_diag.html')

                rend_kwarg_dict["enumerate"] = jinja_enumerate
                rend_kwarg_dict["list"] = jinja_list
                mean_rndr = mean_tmpl.render(rend_kwarg_dict)

                #Write mean diagnostic plots HTML file:
                with open(mean_ptype_file,'w', encoding='utf-8') as ofil:
                    ofil.write(mean_rndr)
                #End with
            #End if (data frame)

            #Also check if index page exists for this case:
            index_html_file = \
                self.__case_web_paths[web_data.case]['website_dir'] / "index.html"
            print("index_html_file",index_html_file)
            
            # Create run info web page
            run_info_md_file = \
                self.__case_web_paths[web_data.case]['website_dir'] / self.run_info
            print("run_info_md_file",run_info_md_file)

            # Read the markdown file
            with open(run_info_md_file, "r", encoding="utf-8") as mdfile:
                md_text = mdfile.read()

            # Convert markdown to HTML
            run_info_html = markdown.markdown(md_text)
            index_title = "AMP Diagnostics Prototype"
            run_info_html_file = self.__case_web_paths[web_data.case]['website_dir'] / "run_info.html"
            run_info_tmpl = jinenv.get_template('template_run_info.html')
            run_info_rndr = run_info_tmpl.render(run_info=run_info_html,
                                                 title=index_title,
                                            case_name=web_data.case,
                                            base_name=data_name,
                                            case_yrs=case_yrs,
                                            baseline_yrs=baseline_yrs,
                                            plot_types=plot_types,
                                            run_info=run_info_html_file)

            with open(run_info_html_file, "w", encoding="utf-8") as htmlfile:
                htmlfile.write(run_info_rndr)

            #Re-et plot types list:
            if web_data.case == 'multi-case':
                plot_types = multi_plot_type_html
            else:
                plot_types = plot_type_html
            plot_types = plot_type_html
            #End if

            #List of ADF default plot types
            avail_plot_types = res["default_ptypes"]
           
            #Check if current plot type is in ADF default.
            #If not, add it so the index.html file can include it
            for ptype in plot_types.keys():
                if ptype not in avail_plot_types:
                    avail_plot_types.append(plot_types)


            # External packages that can be run through ADF
            avail_external_packages = {'MDTF':'mdtf_html_path', 'CVDP':'cvdp_html_path'}
            
            #Construct index.html
            index_title = "AMP Diagnostics Prototype"
            index_tmpl = jinenv.get_template('template_index.html')
            index_rndr = index_tmpl.render(title=index_title,
                                            case_name=web_data.case,
                                            base_name=data_name,
                                            case_yrs=case_yrs,
                                            baseline_yrs=baseline_yrs,
                                            plot_types=plot_types,
                                            avail_plot_types=avail_plot_types,
                                            avail_external_packages=avail_external_packages,
                                            external_package_links=self.external_package_links,
                                            run_info=run_info_html_file)

            #Write Mean diagnostics index HTML file:
            with open(index_html_file, 'w', encoding='utf-8') as ofil:
                ofil.write(index_rndr)
            #End with

        #End for (web data loop)

        #If this is a multi-case instance, then copy website to "main" directory:
        if main_site_path:
            #Add "multi-case" to start of case_names:
            case_names.insert(0, "multi-case")

            #Create CSS templates file path:
            main_templates_path = main_site_path / "templates"

            #loop over cases:
            for case_name in case_names:

                #Check if case name is present in plot
                if case_name in self.__case_web_paths:
                    #Extract website directory:
                    website_dir = self.__case_web_paths[case_name]['website_dir']

                    #Copy website directory to "main site" directory:
                    shutil.copytree(website_dir, main_site_path / case_name)

                    #Also add path to case_sites dictionary:
                    case_sites[case_name] = os.path.join(os.curdir, case_name, "index.html")

                    #Also make sure CSS template files have been copied over:
                    if not main_templates_path.is_dir():
                        css_files_dir = self.__case_web_paths[case_name]['css_files_dir']
                        shutil.copytree(css_files_dir, main_site_path / "templates")
                    #End if
                #End if
            #End for (model case loop)

            #Create multi-case site:
            main_title = "ADF Diagnostics"
            main_tmpl = jinenv.get_template('template_multi_case_index.html')
            main_rndr = main_tmpl.render(title=main_title,
                            case_sites=case_sites,
                            base_name=data_name,
                            baseline_yrs=baseline_yrs,
                            )

            #Write multi-case main HTML file:
            outputfile = main_site_path / "index.html"
            with open(outputfile, 'w', encoding='utf-8') as ofil:
                ofil.write(main_rndr)
            #End with
        #End if

        #Notify user that script has finishedd:
        print("  ...Webpages have been generated successfully.")
#++++++++++++++++++++
#End Class definition
#++++++++++++++++++++