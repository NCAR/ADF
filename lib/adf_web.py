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

        #Initialize website plot type order list:
        self.__plot_type_order = []

        #Set case website path dictionary:
        #--------------------------------
        self.__case_web_paths = {}

        #Extract needed variables from yaml file:
        case_names = self.get_cam_info('cam_case_name', required=True)


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

            #Specify where CSS files will be stored:
            css_files_dir = website_dir / "templates"

            #Add all relevant paths to dictionary for specific case:
            self.__case_web_paths[case_name] = {'website_dir': website_dir,
                                                'img_pages_dir': img_pages_dir,
                                                'assets_dir': assets_dir,
                                                'css_files_dir': css_files_dir}
        #End for
        #--------------------------------


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
            if len(self.get_cam_info("cam_case_name")) > 1:
                case_name = "multi-case"
            else:
                case_name = self.get_cam_info("cam_case_name")[0]
            #End if
        #End if

        #Create HTML file path variable,
        #which will be used in "create_website":
        if not data_frame:
            html_name = f'plot_page_{web_data.stem}.html'
        #End if
        html_file = self.__case_web_paths[case_name]["img_pages_dir"] / html_name

        #Create new variable to store path to image name in assets directory,
        #which will be used in "create_webite":
        if not data_frame:
            asset_path = self.__case_web_paths[case_name]['assets_dir'] / web_data.name
        #End if

        #Initialize web data object:
        web_data = _WebData(web_data, web_name, case_name,
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
        if plot_type not in self.__plot_type_order:
            self.__plot_type_order.append(plot_type)
        #End if

    #########

    def create_website(self):

        """
        Generate webpages to display diagnostic results.
        """

        #import needed standard modules:
        import shutil
        import itertools
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

        #Extract plot_location:
        plot_location = self.plot_location

        #If there is more than one plot location, then create new website directory:
        if len(plot_location) > 1:
            main_site_path = Path(self.get_basic_info('cam_diag_plot_loc', required=True))
            main_site_path = main_site_path / "main_website"
            main_site_path.mkdir(exist_ok=True)
            case_sites = OrderedDict()
        else:
            main_site_path = "" #Set main_site_path to blank value
        #End if

        #Extract needed variables from yaml file:
        case_names = self.get_cam_info('cam_case_name', required=True)

        #Extract variable list:
        var_list = self.diag_var_list

        #Set name of comparison data, which depends on "compare_obs":
        if self.compare_obs:
            data_name = "obs"
        else:
            data_name = self.get_baseline_info('cam_case_name', required=True)
        #End if

        #Extract variable defaults dictionary (for categories):
        var_defaults_dict = self.variable_defaults

        #Set plot type html dictionary (for Jinja templating):
        plot_type_html = OrderedDict()
        for plot_type in self.__plot_type_order:
            if plot_type == 'Tables':
                plot_type_html[plot_type] = os.path.join("html_table", f"mean_table.html")
            else:
                plot_type_html[plot_type] = os.path.join("html_img", f"mean_diag_{plot_type}.html")
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
        #Note: It might be better if the "mean_html_info" dictionary was created in the "add_website_data",
        #so that we only had to do the web_data loop once, but for now this will do. -JN
        mean_html_info = OrderedDict()

        #Loop over all web data objects:
        for web_data in self.__website_data:

            #Create the directory where the website will be built:
            self.__case_web_paths[web_data.case]['website_dir'].mkdir(exist_ok=True)

            #Create a directory that will hold just the html files for individual images:
            self.__case_web_paths[web_data.case]['img_pages_dir'].mkdir(exist_ok=True)

            #Create a directory that will hold copies of the actual images:
            self.__case_web_paths[web_data.case]['assets_dir'].mkdir(exist_ok=True)

            #Create a directory where CSS files will be stored:
            css_files_dir = self.__case_web_paths[web_data.case]['css_files_dir']
            css_files_dir.mkdir(exist_ok=True)

            #Copy CSS files over to output directory:
            for css_file in jinja_template_dir.glob('*.css'):
                shutil.copyfile(css_file, css_files_dir / css_file.name)
            #End for

            if not web_data.data_frame:
                #Variable is a path to an image file, so first extract file name:
                img_name = Path(web_data.data.name)

                #Move file to assets directory:
                shutil.copy(web_data.data, web_data.asset_path)

                #Extract plot_type:
                ptype = web_data.plot_type

                #Initialize Ordered Dictionary for plot type:
                if ptype not in mean_html_info:
                    mean_html_info[ptype] = OrderedDict()
                #End if

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

                if category not in mean_html_info[ptype]:
                    mean_html_info[ptype][category] = OrderedDict()
                #End if

                #Extract web data name (usually the variable name):
                name = web_data.name

                #Initialize Ordered Dictionary for variable:
                if name not in mean_html_info[ptype][category]:
                    mean_html_info[ptype][category][name] = OrderedDict()
                #End if

                #Determine season value:
                if web_data.season:
                    season = web_data.season
                else:
                    season = "plot" #Just have the link be labeled "plot".
                #End if

                #Initialize Ordered Dictionary for season:
                mean_html_info[ptype][category][name][season] = web_data.html_file.name

            #End if (data-frame check)
        #End for (web_data list loop)

        #Loop over all web data objects:
        for web_data in self.__website_data:

            if not web_data.data_frame:

                #Create output HTML file path:
                img_pages_dir = self.__case_web_paths[web_data.case]['img_pages_dir']
                img_data = [os.path.relpath(web_data.asset_path, start=img_pages_dir),
                            web_data.asset_path.stem]

                #Check if plot image already handlse multiple cases:
                if web_data.multi_case:
                    case1 = "Listed in plots."
                else:
                    case1 = web_data.case
                #End if

                #Create titles
                var_title = f"Variable: {web_data.name}"
                season_title = f"Season: {web_data.season}"
                plottype_title = f"Plot: {web_data.plot_type}"
                tmpl = jinenv.get_template('template.html')  #Set template
                rndr = tmpl.render(title=main_title,
                                   var_title=var_title,
                                   season_title=season_title,
                                   plottype_title=plottype_title,
                                   imgs=img_data,
                                   case1=case1,
                                   case2=data_name,
                                   mydata=mean_html_info[web_data.plot_type],
                                   plot_types=plot_type_html) #The template rendered

                #Open HTML file:
                with open(web_data.html_file, 'w', encoding='utf-8') as ofil:
                    ofil.write(rndr)
                #End with

                #Check if the mean plot type page exists for this case:
                mean_ptype_file = img_pages_dir / f"mean_diag_{web_data.plot_type}.html"
                if not mean_ptype_file.exists():

                    #Check if plot image already handlse multiple cases:
                    if web_data.multi_case:
                        case1 = "Listed in plots."
                    else:
                        case1 = web_data.case
                    #End if

                    #Construct individual plot type mean_diag html files, if they don't
                    #already exist:
                    #mean_tmpl = jinenv.get_template(f'template_mean_diag_{web_data.plot_type}.html')
                    mean_tmpl = jinenv.get_template(f'template_mean_diag.html')
                    mean_rndr = mean_tmpl.render(title=main_title,
                                                 case1=case1,
                                                 case2=data_name,
                                                 mydata=mean_html_info[web_data.plot_type],
                                                 curr_type=web_data.plot_type,
                                                 plot_types=plot_type_html)

                    #Write mean diagnostic plots HTML file:
                    with open(mean_ptype_file,'w', encoding='utf-8') as ofil:
                        ofil.write(mean_rndr)
                    #End with
                #End if (mean_ptype exists)

                #Also check if index page exists for this case:
                index_html_file = \
                    self.__case_web_paths[web_data.case]['website_dir'] / "index.html"

                if not index_html_file.exists():

                    #Construct index.html
                    index_title = "AMP Diagnostics Prototype"
                    index_tmpl = jinenv.get_template('template_index.html')
                    index_rndr = index_tmpl.render(title=main_title,
                                                   case1=web_data.case,
                                                   case2=data_name,
                                                   #gen_table_html=gen_table_html,
                                                   plot_types=plot_type_html)

                    #Write Mean diagnostics index HTML file:
                    with open(index_html_file, 'w', encoding='utf-8') as ofil:
                        ofil.write(index_rndr)
                    #End with
                #End if (mean_index exists)
            #End if (data frame)
        #End for (web data loop)

        #Notify user that script has finishedd:
        print("  ...Webpages have been generated successfully.")

        #EVERYTHING BELOW HERE IS TEMPORARILY COMMENTED OUT  FOR TESTING -JN !!!!!!!!!!!!!!!!!!!
        #UNCOMMENT ONCE CODE IS READY FOR DATA FRAMES!!!!!!!!!!!!!!!!!

'''

        #Grab AMWG Table HTML files:
        table_html_files = list(plot_path.glob(f"amwg_table_{case_name}*.html"))

        #Grab the comparison table and move it to website dir
        comp_table_html_file = list(plot_path.glob("*comp.html"))

        #Also grab baseline/obs tables, which are always stored in the first case directory:
        if case_idx == 0:
            data_table_html_files = list(plot_path.glob(f"amwg_table_{data_name}*.html"))
        #End if

        #Determine if any AMWG tables were generated:
        if table_html_files:

        #Set Table HTML generation logical to "TRUE":
        gen_table_html = True

                #Create a directory that will hold table html files:
                table_pages_dir = website_dir / "html_table"
                table_pages_dir.mkdir(exist_ok=True)

                #Move all case table html files to new directory:
                for table_html in table_html_files:
                    shutil.move(table_html, table_pages_dir / table_html.name)
                #End for

                #copy all data table html files as well:
                for data_table_html in data_table_html_files:
                    shutil.copy2(data_table_html, table_pages_dir / data_table_html.name)
                #End for

                #Construct dictionary needed for HTML page:
                amwg_tables = OrderedDict()

                for case in [case_name, data_name]:

                    #Search for case name in moved HTML files:
                    table_htmls = sorted(table_pages_dir.glob(f"amwg_table_{case}.html"))

                    #Check if file exists:
                    if table_htmls:

                        #Initialize loop counter:
                        count = 0

                        #Loop over globbed files:
                        for table_html in table_htmls:

                            #Create relative path for HTML file:
                            amwg_tables[case] = table_html.name

                            #Update counter:
                            count += 1

                            #If counter greater than one, then throw an error:
                            if count > 1:
                                emsg = f"More than one AMWG table is associated with case '{case}'."
                                emsg += "\nNot sure what is going on, "
                                emsg += "\nso website generation will end here."
                                self.end_diag_fail(emsg)
                            #End if
                        #End for (table html file loop)
                    #End if (table html file exists check)
                #End for (case vs data)

                #Check if comp table exists
                #(if not, then obs are being compared and comp table is not created)
                if comp_table_html_file:
                    #Move the comparison table html file to new directory
                    for comp_table in comp_table_html_file:
                        shutil.move(comp_table, table_pages_dir / comp_table.name)
                        #Add comparison table to website dictionary
                        # * * * * * * * * * * * * * * * * * * * * * * * * * * * *
                        #This will be for single-case for now,
                        #will need to think how to change as multi-case is introduced
                        amwg_tables["Case Comparison"] = comp_table.name
                        # * * * * * * * * * * * * * * * * * * * * * * * * * * * *
                    #End for

                # need this to grab the locations of the amwg tables...
                amwg_table_data = [str(table_pages_dir / table_html.name), ""]

                #Construct mean_table.html
                mean_tmpl = jinenv.get_template('template_mean_table.html')
                mean_rndr = mean_tmpl.render(title=main_title,
                                value=amwg_table_data,
                                case1=case_name,
                                case2=data_name,
                                amwg_tables=amwg_tables,
                                plot_types=plot_type_html,
                                )

                #Write mean diagnostic tables HTML file:
                outputfile = table_pages_dir / "mean_table.html"
                with open(outputfile, 'w', encoding='utf-8') as ofil:
                    ofil.write(mean_rndr)
                #End with
            else:
                #No Tables exist, so no link will be added to main page:
                gen_table_html = False
            #End if

'''

        #Construct index.html
        #index_title = "AMP Diagnostics Prototype"
        #index_tmpl = jinenv.get_template('template_index.html')
        #index_rndr = index_tmpl.render(title=main_title,
        #                 case1=case_name,
        #                 case2=data_name,
        #                 gen_table_html=gen_table_html,
        #                 plot_types=plot_type_html,
        #                 )

        #Write Mean diagnostics HTML file:
        #outputfile = website_dir / "index.html"
        #with open(outputfile, 'w', encoding='utf-8') as ofil:
        #    ofil.write(index_rndr)
        #End with

        #IGNORE MULTI-CASE INFO FOR NOW!!!!!!!!!!!!!!!!! -JN

'''

        #If this is a multi-case instance, then copy website to "main" directory:
            if main_site_path:
                shutil.copytree(website_dir, main_site_path / case_name)
                #Also add path to case_sites dictionary:
                case_sites[case_name] = os.path.join(os.curdir, case_name, "index.html")
                #Finally, if first case, then also copy templates directory for CSS files:
                if case_idx == 0:
                    shutil.copytree(css_files_dir, main_site_path / "templates")
                #End if
            #End if
        #End for (model case loop)

        #Create multi-case site, if needed:
        if main_site_path:
            main_title = "ADF Diagnostics"
            main_tmpl = jinenv.get_template('template_multi_case_index.html')
            main_rndr = main_tmpl.render(title=main_title,
                            case_sites=case_sites,
                            )
            #Write multi-case main HTML file:
            outputfile = main_site_path / "index.html"
            with open(outputfile, 'w', encoding='utf-8') as ofil:
                ofil.write(main_rndr)
            #End with
        #End if

'''

#++++++++++++++++++++
#End Class definition
#++++++++++++++++++++
